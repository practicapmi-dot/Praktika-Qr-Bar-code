#!/usr/bin/env python3
"""
Уникальные бинаризованные кропы кодов с видео: YOLO + ByteTrack.

Задача: один физический код детектится в десятках кадров подряд — кропать его
нужно ОДИН раз за появление в кадре. Трекер (ByteTrack, встроен в ultralytics)
присваивает боксу track_id; пока объект в кадре, копим его лучший по резкости
кроп (max lap_var). Когда трек пропал на --gone-after обработанных кадров (или
видео кончилось) — кроп бинаризуется (ч/б по порогу --bin-thr) и коммитится в
массив. Объект ушёл из кадра и вернулся → трекер даёт НОВЫЙ id → новый кроп.
Треки, ни разу не прошедшие фильтр резкости (--blur-thr и т.д. из
infer_video.py), отбрасываются (вернуть их — флаг --keep-blurry).

Выход (--out):
  crops.npz       — массив ч/б кропов uint8 (0/255), ключи crop_00000…
  crops_meta.json — по индексу: video, track_id, cls, frame, lap_var, xyxy
  crops/*.png     — те же кропы картинками (посмотреть глазами)
  *_annot.mp4     — видео с track id; закоммиченный трек помечается CAP

Перебор кропов в своём коде:
  data = np.load("runs/track_crops/crops.npz")
  crops = [data[k] for k in data.files]           # список ч/б numpy-массивов
  meta = json.load(open("runs/track_crops/crops_meta.json"))

Либо импортом (запуск из src/):
  from track_crops import collect_crops

Нюанс: при --stride 5 (20fps → 4fps) ByteTrack держит потерянный трек
track_buffer=30 обработанных кадров ≈ 7.5 c реального времени — объект,
вернувшийся быстрее, повторно не кропается (то же появление). Если появляются
дубли из-за ID-switch — уменьшить --stride до 2–3.

Пример:
  .venv/bin/python src/track_crops.py --weights best.pt --source OZONVIDEOS \
      --out runs/track_crops --conf 0.25 --imgsz 640 --stride 5
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from infer_video import CLASS_NAMES, COLORS, crop_sharpness, extract_crop, list_videos  # noqa: E402


def build_parser():
    p = argparse.ArgumentParser(description="Unique binarized code crops via YOLO + ByteTrack.")
    p.add_argument("--weights", default="best.pt")
    p.add_argument("--source", required=True, help="видеофайл или папка с видео")
    p.add_argument("--out", default="runs/track_crops")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--stride", type=int, default=5, help="обрабатывать каждый N-й кадр")
    p.add_argument("--pad", type=float, default=0.10,
                   help="запас вокруг BB при кропе, доля от размера бокса "
                        "(0.10 = +10%% с каждой стороны)")
    p.add_argument("--max-frames", type=int, default=0, help="лимит обработанных кадров (0=все)")
    p.add_argument("--bin-thr", type=int, default=210,
                   help="порог бинаризации: >thr → белый(255), иначе чёрный(0); <=0 → Otsu")
    p.add_argument("--blur-thr", type=float, default=1500.0,
                   help="мин. резкость лучшего кропа трека (дисперсия Лапласиана)")
    p.add_argument("--min-contrast", type=float, default=25.0)
    p.add_argument("--min-box", type=int, default=24, help="мин. сторона бокса, px")
    p.add_argument("--gone-after", type=int, default=15,
                   help="через сколько обработанных кадров без трека коммитить его кроп")
    p.add_argument("--keep-blurry", action="store_true",
                   help="коммитить и треки, не прошедшие фильтр резкости")
    p.add_argument("--no-video", action="store_true", help="не писать аннотированное видео")
    p.add_argument("--annot-width", type=int, default=1280, help="ширина аннотир. видео")
    p.add_argument("--device", default="cpu")
    return p


def parse_args():
    return build_parser().parse_args()


def binarize(crop_bgr, thr=210):
    """BGR/gray-кроп → ч/б uint8 (0/255): pixel > thr → 255 (фон), иначе 0 (штрих).

    thr <= 0 → авто-порог Otsu (устойчивее при неравномерной освещённости).
    """
    g = crop_bgr if crop_bgr.ndim == 2 else cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    if thr <= 0:
        _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, bw = cv2.threshold(g, thr, 255, cv2.THRESH_BINARY)
    return bw


def collect_crops(video_path, a, model=None):
    """Прогнать одно видео, вернуть (crops, raws, meta, stats).

    crops — list[np.ndarray]: бинаризованные ч/б кропы, один на трек;
    raws  — list[np.ndarray] той же длины: те же кропы сырыми (BGR), для
            даунстрим-обработки (нормализация и т.п.);
    meta  — list[dict] той же длины: {video, track_id, cls, frame, lap_var, xyxy};
    stats — {tracks, committed, dropped_blurry, frames}.
    """
    from ultralytics import YOLO
    if model is None:
        model = YOLO(a.weights)  # свежий инстанс = чистое состояние трекера

    vp = Path(video_path)
    cap = cv2.VideoCapture(str(vp))
    if not cap.isOpened():
        print(f"! не открыть {vp.name}")
        return [], [], [], {}
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    ow = oh = 0
    if not a.no_video and W > 0:
        ow = min(a.annot_width, W); oh = int(H * ow / W)
        out = Path(a.out)
        writer = cv2.VideoWriter(str(out / (vp.stem + "_annot.mp4")),
                                 cv2.VideoWriter_fourcc(*"mp4v"),
                                 max(1.0, fps / a.stride), (ow, oh))

    active = {}     # tid -> лучший кандидат {crop, lap, contrast, frame, xyxy, cls, sharp}
    misses = {}     # tid -> обработанных кадров подряд без этого трека
    committed = set()
    crops, raws, meta = [], [], []
    stats = {"tracks": 0, "committed": 0, "dropped_blurry": 0, "frames": 0}

    def commit(tid):
        cand = active.pop(tid)
        misses.pop(tid, None)
        committed.add(tid)
        if not cand["sharp"] and not a.keep_blurry:
            stats["dropped_blurry"] += 1
            return
        crops.append(binarize(cand["crop"], a.bin_thr))
        raws.append(cand["crop"])
        meta.append({"video": vp.name, "track_id": tid,
                     "cls": CLASS_NAMES.get(cand["cls"], cand["cls"]),
                     "frame": cand["frame"], "lap_var": round(cand["lap"], 1),
                     "xyxy": [round(x, 1) for x in cand["xyxy"]]})
        stats["committed"] += 1

    fi = -1
    t0 = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        fi += 1
        if fi % a.stride:
            continue
        stats["frames"] += 1
        res = model.track(frame, persist=True, conf=a.conf, imgsz=a.imgsz,
                          device=a.device, verbose=False)[0]
        boxes = res.boxes
        seen = set()
        vis = frame.copy() if writer is not None else None
        for b in boxes:
            if b.id is None:  # трекер бокс ещё не подтвердил
                continue
            tid = int(b.id); cid = int(b.cls); xyxy = b.xyxy[0].tolist()
            seen.add(tid)
            if tid not in committed:
                crop = extract_crop(frame, xyxy, pad=a.pad)
                if crop is not None:
                    lap, contrast = crop_sharpness(crop)
                    bw_, bh_ = xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]
                    sharp = (lap >= a.blur_thr and contrast >= a.min_contrast
                             and min(bw_, bh_) >= a.min_box)
                    if tid not in active:
                        stats["tracks"] += 1
                    if tid not in active or lap > active[tid]["lap"]:
                        active[tid] = {"crop": crop, "lap": lap, "contrast": contrast,
                                       "frame": fi, "xyxy": xyxy, "cls": cid, "sharp": sharp}
                misses[tid] = 0
            if vis is not None:
                x1, y1, x2, y2 = map(int, xyxy)
                col = COLORS.get(cid, (255, 255, 255)) if tid not in committed else (128, 128, 128)
                label = f"id {tid}" + (" CAP" if tid in committed else "")
                cv2.rectangle(vis, (x1, y1), (x2, y2), col, 3)
                cv2.putText(vis, label, (x1, max(12, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, col, 2)
        for tid in list(misses):
            if tid in seen:
                continue
            misses[tid] += 1
            if misses[tid] >= a.gone_after:
                commit(tid)
        if writer is not None:
            writer.write(cv2.resize(vis, (ow, oh)))
        if a.max_frames and stats["frames"] >= a.max_frames:
            break
    for tid in list(active):  # конец видео — коммит всех живых треков
        commit(tid)
    cap.release()
    if writer is not None:
        writer.release()
    dt = time.time() - t0
    print(f"[{vp.name}] кадров {stats['frames']} ({stats['frames'] / max(1e-6, dt):.2f} fps)"
          f" | треков {stats['tracks']} | закропано {stats['committed']}"
          f" | отброшено blurry {stats['dropped_blurry']}")
    return crops, raws, meta, stats


def run(a):
    """Полный проход по видео: собрать кропы + сохранить все артефакты в a.out.

    Возвращает (all_crops, all_raws, all_meta) — бинаризованные кропы, сырые
    BGR-кропы и метаданные (индексы согласованы).
    """
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    png_dir = out / "crops"
    png_dir.mkdir(exist_ok=True)
    videos = list_videos(a.source)
    if not videos:
        raise SystemExit(f"нет видео в {a.source}")

    all_crops, all_raws, all_meta = [], [], []
    for vp in videos:
        crops, raws, meta, _ = collect_crops(vp, a)
        all_crops += crops
        all_raws += raws
        all_meta += meta

    for i, m in enumerate(all_meta):
        m["idx"] = i
        cv2.imwrite(str(png_dir / f"{i:05d}_track{m['track_id']}_{m['cls']}.png"), all_crops[i])
    np.savez_compressed(out / "crops.npz",
                        **{f"crop_{i:05d}": c for i, c in enumerate(all_crops)})
    (out / "crops_meta.json").write_text(
        json.dumps(all_meta, ensure_ascii=False, indent=2))

    print(f"\nИтого уникальных кропов: {len(all_crops)}")
    print(f"Массив: {out / 'crops.npz'} (np.load → [data[k] for k in data.files])")
    print(f"Мета:   {out / 'crops_meta.json'}")
    print(f"PNG:    {png_dir}/")
    if not a.no_video:
        print(f"Видео:  {out}/*_annot.mp4")
    return all_crops, all_raws, all_meta


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
