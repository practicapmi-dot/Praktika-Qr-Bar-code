#!/usr/bin/env python3
"""
Инференс YOLO-детектора кодов на видео + операционные метрики.

ВАЖНО: на сыром видео НЕТ ground-truth разметки, поэтому mAP/precision/recall
посчитать невозможно (не с чем сравнивать). Считаем ОПЕРАЦИОННЫЕ метрики:
  - детекции по классам и средняя уверенность;
  - coverage — доля кадров, где найден хотя бы один код;
  - decode success-rate — доля найденных кодов, которые реально декодируются
    (zxing-cpp); это ключевая end-to-end метрика «работает ли пайплайн».

Пример:
  .venv/bin/python src/infer_video.py --weights best.pt --source OZONVIDEOS \
      --out runs/video --conf 0.25 --imgsz 640 --stride 5 --decode
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
    p.add_argument("--no-video", action="store_true", help="не писать аннотированное видео")
    p.add_argument("--annot-width", type=int, default=1280, help="ширина аннотир. видео (даунскейл)")
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def list_videos(src):
    s = Path(src)
    if s.is_file():
        return [s]
    return sorted(f for f in s.iterdir() if f.suffix.lower() in VIDEO_EXTS)


def decode_crop(zx, frame, xyxy, pad=0.10):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = xyxy
    bw, bh = x2 - x1, y2 - y1
    x1 = max(0, int(x1 - pad * bw)); y1 = max(0, int(y1 - pad * bh))
    x2 = min(w, int(x2 + pad * bw)); y2 = min(h, int(y2 + pad * bh))
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
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
              "stride": a.stride, "decode": bool(zx), "videos": []}
    G = {"cls": Counter(), "conf": defaultdict(list), "proc": 0, "with": 0,
         "det": 0, "dec": 0, "att": 0, "uniq": set()}

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
             "det": 0, "dec": 0, "att": 0, "uniq": set()}
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
                if zx is not None:
                    v["att"] += 1
                    dec = decode_crop(zx, frame, xyxy)
                    if dec:
                        v["dec"] += 1; v["uniq"].add(dec[0])
                if vis is not None:
                    x1, y1, x2, y2 = map(int, xyxy); col = COLORS.get(cid, (255, 255, 255))
                    cv2.rectangle(vis, (x1, y1), (x2, y2), col, 3)
                    cv2.putText(vis, f"{CLASS_NAMES.get(cid, cid)} {cf:.2f}",
                                (x1, max(12, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, col, 2)
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
        }
        if zx is not None:
            summ["decode"] = {"attempts": v["att"], "decoded": v["dec"],
                              "decode_rate_pct": round(100 * v["dec"] / max(1, v["att"]), 1),
                              "unique_values": len(v["uniq"])}
        report["videos"].append(summ)
        line = (f"[{vp.name}] кадров {v['proc']} | детекций {v['det']} "
                f"{dict(summ['det_per_class'])} | coverage {summ['coverage_pct']}%")
        if zx is not None:
            line += (f" | decode {v['dec']}/{v['att']} ({summ['decode']['decode_rate_pct']}%)"
                     f" uniq {len(v['uniq'])}")
        print(line)

        G["proc"] += v["proc"]; G["with"] += v["with"]; G["det"] += v["det"]
        G["dec"] += v["dec"]; G["att"] += v["att"]; G["uniq"] |= v["uniq"]
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
    }
    if G["att"]:
        agg["decode"] = {"attempts": G["att"], "decoded": G["dec"],
                         "decode_rate_pct": round(100 * G["dec"] / max(1, G["att"]), 1),
                         "unique_values": len(G["uniq"])}
    report["aggregate"] = agg
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))

    print("\n=== ИТОГО (агрегат по всем видео) ===")
    print(json.dumps(agg, ensure_ascii=False, indent=2))
    print(f"\nОтчёт: {out / 'report.json'}")
    if not a.no_video:
        print(f"Аннотированные видео: {out}/*_annot.mp4")


if __name__ == "__main__":
    main()
