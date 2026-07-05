#!/usr/bin/env python3
"""
Инференс YOLO-детектора кодов на видео + операционные метрики.

ВАЖНО: на сыром видео НЕТ ground-truth разметки, поэтому mAP/precision/recall
посчитать невозможно (не с чем сравнивать). Считаем ОПЕРАЦИОННЫЕ метрики:
  - детекции по классам и средняя уверенность;
  - coverage — доля кадров, где найден хотя бы один код;
  - decode success-rate — доля найденных кодов, которые реально декодируются
    (zxing-cpp); это ключевая end-to-end метрика «работает ли пайплайн».

Фильтр качества кропа (sharp / blurry):
  Каждый кроп оцениваем дисперсией Лапласиана (резкость) + контрастом (std
  яркости) + минимальной стороной бокса. Кроп считается «sharp», если
  laplacian_var >= --blur-thr И contrast >= --min-contrast И
  min(w,h) >= --min-box. Размытые рисуются серым с меткой BLUR.
  Калибровка порога: запустить с --log-dets, из detections.jsonl вытащить кропы
  по диапазонам lap_var и посмотреть глазами (или, если коды декодируются,
  добавить --decode и смотреть decode.by_quality в report.json — у sharp
  decode-rate должен быть сильно выше). Дефолт --blur-thr=1500 откалиброван
  визуально по OZONVIDEOS (4K @ 20fps, motion blur): ниже ~1500 штрихи смазаны,
  выше — различимы.

Пример:
  .venv/bin/python src/infer_video.py --weights best.pt --source OZONVIDEOS \
      --out runs/video --conf 0.25 --imgsz 640 --stride 5 --log-dets
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv"}
CLASS_NAMES = {0: "qr", 1: "barcode_1d", 2: "datamatrix", 3: "pdf417", 4: "aztec"}
COLORS = {0: (0, 200, 0), 1: (255, 128, 0), 2: (0, 0, 255), 3: (255, 0, 255), 4: (0, 200, 200)}


def parse_args():
    p = argparse.ArgumentParser(description="YOLO inference on video + metrics.")
    p.add_argument("--weights", default="best.pt")
    p.add_argument("--source", required=True, help="видеофайл или папка с видео")
    p.add_argument("--out", default="runs/video")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--stride", type=int, default=5, help="обрабатывать каждый N-й кадр")
    p.add_argument("--max-frames", type=int, default=0, help="лимит обработанных кадров (0=все)")
    p.add_argument("--decode", action="store_true", help="считать decode-rate (zxing-cpp)")
    p.add_argument("--blur-thr", type=float, default=1500.0,
                   help="порог резкости (дисперсия Лапласиана); ниже — blurry; "
                        "дефолт откалиброван визуально по OZONVIDEOS")
    p.add_argument("--min-contrast", type=float, default=25.0,
                   help="мин. контраст кропа (std яркости); ниже — blurry")
    p.add_argument("--min-box", type=int, default=24,
                   help="мин. сторона бокса, px; меньше — blurry")
    p.add_argument("--only-sharp", action="store_true",
                   help="декодировать только sharp-кропы (blurry пропускаются)")
    p.add_argument("--log-dets", action="store_true",
                   help="писать detections.jsonl с метриками каждого кропа")
    p.add_argument("--no-video", action="store_true", help="не писать аннотированное видео")
    p.add_argument("--annot-width", type=int, default=1280, help="ширина аннотир. видео (даунскейл)")
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def list_videos(src):
    s = Path(src)
    if s.is_file():
        return [s]
    return sorted(f for f in s.iterdir() if f.suffix.lower() in VIDEO_EXTS)


def extract_crop(frame, xyxy, pad=0.10):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = xyxy
    bw, bh = x2 - x1, y2 - y1
    x1 = max(0, int(x1 - pad * bw)); y1 = max(0, int(y1 - pad * bh))
    x2 = min(w, int(x2 + pad * bw)); y2 = min(h, int(y2 + pad * bh))
    crop = frame[y1:y2, x1:x2]
    return crop if crop.size else None


def crop_sharpness(crop, norm_side=256):
    """Резкость (дисперсия Лапласиана) + контраст (std яркости) кропа.

    Кроп даунскейлится до norm_side по большей стороне, чтобы порог не зависел
    от размера бокса (апскейл не делаем — информации он не добавляет).
    """
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    h, w = g.shape
    scale = norm_side / max(h, w)
    if scale < 1.0:
        g = cv2.resize(g, (max(1, int(w * scale)), max(1, int(h * scale))),
                       interpolation=cv2.INTER_AREA)
    lap_var = float(cv2.Laplacian(g, cv2.CV_64F).var())
    contrast = float(g.std())
    return lap_var, contrast


def decode_crop(zx, crop):
    try:
        for r in zx.read_barcodes(crop):
            if r.text:
                return r.text, str(r.format)
    except Exception:
        return None
    return None


def main():
    a = parse_args()
    from ultralytics import YOLO

    zx = None
    if a.decode:
        try:
            import zxingcpp as zx  # noqa: F401
        except ImportError:
            print("! zxing-cpp не установлен — метрика декодирования пропущена")
            zx = None

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    model = YOLO(a.weights)
    videos = list_videos(a.source)
    if not videos:
        raise SystemExit(f"нет видео в {a.source}")

    report = {"weights": a.weights, "conf": a.conf, "imgsz": a.imgsz,
              "stride": a.stride, "decode": bool(zx),
              "quality_filter": {"blur_thr": a.blur_thr, "min_contrast": a.min_contrast,
                                 "min_box": a.min_box, "only_sharp": a.only_sharp},
              "videos": []}
    QK = ["sharp", "blur", "att_sharp", "dec_sharp", "att_blur", "dec_blur"]
    G = {"cls": Counter(), "conf": defaultdict(list), "proc": 0, "with": 0,
         "det": 0, "dec": 0, "att": 0, "uniq": set(), **{k: 0 for k in QK}}
    det_log = open(out / "detections.jsonl", "w") if a.log_dets else None

    for vp in videos:
        cap = cv2.VideoCapture(str(vp))
        if not cap.isOpened():
            print(f"! не открыть {vp.name}")
            continue
        fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = None
        ow = oh = 0
        if not a.no_video and W > 0:
            ow = min(a.annot_width, W); oh = int(H * ow / W)
            writer = cv2.VideoWriter(str(out / (vp.stem + "_annot.mp4")),
                                     cv2.VideoWriter_fourcc(*"mp4v"),
                                     max(1.0, fps / a.stride), (ow, oh))
        v = {"cls": Counter(), "conf": defaultdict(list), "proc": 0, "with": 0,
             "det": 0, "dec": 0, "att": 0, "uniq": set(), **{k: 0 for k in QK}}
        fi = -1; t0 = time.time()
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            fi += 1
            if fi % a.stride:
                continue
            v["proc"] += 1
            res = model.predict(frame, conf=a.conf, imgsz=a.imgsz,
                                device=a.device, verbose=False)[0]
            boxes = res.boxes
            if len(boxes):
                v["with"] += 1
            vis = frame.copy() if writer is not None else None
            for b in boxes:
                cid = int(b.cls); cf = float(b.conf); xyxy = b.xyxy[0].tolist()
                v["cls"][cid] += 1; v["conf"][cid].append(cf); v["det"] += 1
                crop = extract_crop(frame, xyxy)
                lap = contrast = 0.0
                sharp = False
                if crop is not None:
                    lap, contrast = crop_sharpness(crop)
                    bw, bh = xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]
                    sharp = (lap >= a.blur_thr and contrast >= a.min_contrast
                             and min(bw, bh) >= a.min_box)
                q = "sharp" if sharp else "blur"
                v[q] += 1
                dec = None
                if zx is not None and crop is not None and (sharp or not a.only_sharp):
                    v["att"] += 1; v["att_" + q] += 1
                    dec = decode_crop(zx, crop)
                    if dec:
                        v["dec"] += 1; v["dec_" + q] += 1; v["uniq"].add(dec[0])
                if det_log is not None:
                    det_log.write(json.dumps({
                        "video": vp.name, "frame": fi, "cls": CLASS_NAMES.get(cid, cid),
                        "conf": round(cf, 3), "xyxy": [round(x, 1) for x in xyxy],
                        "lap_var": round(lap, 1), "contrast": round(contrast, 1),
                        "sharp": sharp, "decoded": bool(dec),
                        "text": dec[0] if dec else None}, ensure_ascii=False) + "\n")
                if vis is not None:
                    x1, y1, x2, y2 = map(int, xyxy)
                    col = COLORS.get(cid, (255, 255, 255)) if sharp else (128, 128, 128)
                    label = f"{CLASS_NAMES.get(cid, cid)} {cf:.2f}"
                    if not sharp:
                        label += " BLUR"
                    if dec:
                        label += " OK"
                    cv2.rectangle(vis, (x1, y1), (x2, y2), col, 3)
                    cv2.putText(vis, label, (x1, max(12, y1 - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, col, 2)
            if writer is not None:
                writer.write(cv2.resize(vis, (ow, oh)))
            if a.max_frames and v["proc"] >= a.max_frames:
                break
        dt = time.time() - t0
        cap.release()
        if writer is not None:
            writer.release()

        summ = {
            "video": vp.name, "resolution": f"{W}x{H}", "fps": round(fps, 1),
            "frames_total_est": total, "frames_processed": v["proc"], "stride": a.stride,
            "frames_with_detections": v["with"],
            "coverage_pct": round(100 * v["with"] / max(1, v["proc"]), 1),
            "detections_total": v["det"],
            "det_per_class": {CLASS_NAMES[c]: v["cls"][c] for c in sorted(v["cls"])},
            "mean_conf_per_class": {CLASS_NAMES[c]: round(float(np.mean(v["conf"][c])), 3)
                                    for c in sorted(v["conf"])},
            "proc_fps": round(v["proc"] / max(1e-6, dt), 2),
            "quality": {"sharp": v["sharp"], "blurry": v["blur"],
                        "sharp_pct": round(100 * v["sharp"] / max(1, v["det"]), 1)},
        }
        if zx is not None:
            summ["decode"] = {"attempts": v["att"], "decoded": v["dec"],
                              "decode_rate_pct": round(100 * v["dec"] / max(1, v["att"]), 1),
                              "unique_values": len(v["uniq"]),
                              "by_quality": {
                                  "sharp": {"attempts": v["att_sharp"], "decoded": v["dec_sharp"],
                                            "rate_pct": round(100 * v["dec_sharp"] / max(1, v["att_sharp"]), 1)},
                                  "blurry": {"attempts": v["att_blur"], "decoded": v["dec_blur"],
                                             "rate_pct": round(100 * v["dec_blur"] / max(1, v["att_blur"]), 1)}}}
        report["videos"].append(summ)
        line = (f"[{vp.name}] кадров {v['proc']} | детекций {v['det']} "
                f"{dict(summ['det_per_class'])} | coverage {summ['coverage_pct']}%"
                f" | sharp {v['sharp']}/{v['det']} ({summ['quality']['sharp_pct']}%)")
        if zx is not None:
            line += (f" | decode {v['dec']}/{v['att']} ({summ['decode']['decode_rate_pct']}%)"
                     f" [sharp {v['dec_sharp']}/{v['att_sharp']},"
                     f" blur {v['dec_blur']}/{v['att_blur']}]"
                     f" uniq {len(v['uniq'])}")
        print(line)

        G["proc"] += v["proc"]; G["with"] += v["with"]; G["det"] += v["det"]
        G["dec"] += v["dec"]; G["att"] += v["att"]; G["uniq"] |= v["uniq"]
        for k in QK:
            G[k] += v[k]
        for c in v["cls"]:
            G["cls"][c] += v["cls"][c]
        for c in v["conf"]:
            G["conf"][c] += v["conf"][c]

    agg = {
        "videos": len(report["videos"]), "frames_processed": G["proc"],
        "frames_with_detections": G["with"],
        "coverage_pct": round(100 * G["with"] / max(1, G["proc"]), 1),
        "detections_total": G["det"],
        "det_per_class": {CLASS_NAMES[c]: G["cls"][c] for c in sorted(G["cls"])},
        "mean_conf_per_class": {CLASS_NAMES[c]: round(float(np.mean(G["conf"][c])), 3)
                                for c in sorted(G["conf"])},
        "quality": {"sharp": G["sharp"], "blurry": G["blur"],
                    "sharp_pct": round(100 * G["sharp"] / max(1, G["det"]), 1)},
    }
    if G["att"]:
        agg["decode"] = {"attempts": G["att"], "decoded": G["dec"],
                         "decode_rate_pct": round(100 * G["dec"] / max(1, G["att"]), 1),
                         "unique_values": len(G["uniq"]),
                         "by_quality": {
                             "sharp": {"attempts": G["att_sharp"], "decoded": G["dec_sharp"],
                                       "rate_pct": round(100 * G["dec_sharp"] / max(1, G["att_sharp"]), 1)},
                             "blurry": {"attempts": G["att_blur"], "decoded": G["dec_blur"],
                                        "rate_pct": round(100 * G["dec_blur"] / max(1, G["att_blur"]), 1)}}}
    report["aggregate"] = agg
    if det_log is not None:
        det_log.close()
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))

    print("\n=== ИТОГО (агрегат по всем видео) ===")
    print(json.dumps(agg, ensure_ascii=False, indent=2))
    print(f"\nОтчёт: {out / 'report.json'}")
    if det_log is not None:
        print(f"Лог детекций (для калибровки --blur-thr): {out / 'detections.jsonl'}")
    if not a.no_video:
        print(f"Аннотированные видео: {out}/*_annot.mp4")


if __name__ == "__main__":
    main()
