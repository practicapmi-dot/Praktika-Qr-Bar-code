#!/usr/bin/env python3
"""
Объединённый пайплайн: track_crops (детекция+трекинг) → barcode_normalizer.

Этап 1 — track_crops.run(): YOLO + ByteTrack собирают по одному лучшему кропу
на появление объекта в кадре (+ бинаризованный массив crops.npz, как раньше).
Этап 2 — каждый СЫРОЙ кроп из массива прогоняется через barcode_normalizer
(выпрямление 1D-кода: ориентация штрихов → поворот → сегментация зоны →
гомография → апскейл) и результат сохраняется в отдельную папку normalized/.

Нормализатор рассчитан на 1D-штрихкоды, поэтому по умолчанию нормализуются
только кропы класса barcode_1d (--norm-classes); кропы остальных классов
кладутся в normalized/ как есть, чтобы папка была полной.

Выход (--out, default runs/pipeline):
  crops.npz, crops_meta.json, crops/*.png, *_annot.mp4 — всё из track_crops
  normalized/*.png  — выпрямленные кропы (индексы совпадают с crops/)
  normalized.npz    — те же нормализованные изображения массивом

Пример:
  .venv/bin/python src/pipeline_video.py --weights best.pt --source OZONVIDEOS \
      --out runs/pipeline --stride 5 --bin-thr 0
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from track_crops import build_parser, run  # noqa: E402
from barcode_normalizer import NormalizerConfig, normalize_barcode  # noqa: E402


def parse_args():
    p = build_parser()
    p.description = "Video → tracked unique crops → barcode normalizer."
    p.set_defaults(out="runs/pipeline")
    p.add_argument("--target-height", type=int, default=256,
                   help="высота выпрямленного кода после нормализации")
    p.add_argument("--norm-gray", action="store_true",
                   help="нормализованный выход одноканальный (grayscale)")
    p.add_argument("--no-perspective", action="store_true",
                   help="отключить коррекцию перспективы в нормализаторе")
    p.add_argument("--norm-classes", default="barcode_1d",
                   help="какие классы нормализовать (через запятую); "
                        "остальные копируются как есть")
    return p.parse_args()


def normalize_all(raws, meta, a):
    """Прогнать сырые кропы через нормализатор.

    Возвращает list той же длины, что raws: выпрямленное изображение, либо
    исходный кроп (класс вне --norm-classes), либо None (нормализатор упал).
    """
    cfg = NormalizerConfig(target_height=a.target_height,
                           output_grayscale=a.norm_gray,
                           correct_perspective=not a.no_perspective)
    norm_classes = {c.strip() for c in a.norm_classes.split(",") if c.strip()}
    out, failed = [], 0
    for raw, m in zip(raws, meta):
        if m["cls"] not in norm_classes:
            out.append(raw)
            continue
        h, w = raw.shape[:2]
        try:
            norm = normalize_barcode(raw, (0, 0, w, h), cfg)
        except Exception as e:
            print(f"! normalizer упал на idx {m['idx']} (track {m['track_id']}): {e}")
            norm = None
        if norm is None:
            failed += 1
        out.append(norm)
    if failed:
        print(f"! нормализатор не справился с {failed} кропами — "
              f"их нет в normalized/ (индексы сохранены)")
    return out


def main():
    a = parse_args()

    # Этап 1: трекинг и сбор уникальных кропов (+ все артефакты track_crops).
    crops, raws, meta = run(a)

    # Этап 2: нормализация сырых кропов в отдельную папку.
    norm_dir = Path(a.out) / "normalized"
    norm_dir.mkdir(exist_ok=True)
    normed = normalize_all(raws, meta, a)

    saved = {}
    for m, img in zip(meta, normed):
        if img is None:
            continue
        name = f"{m['idx']:05d}_track{m['track_id']}_{m['cls']}"
        cv2.imwrite(str(norm_dir / (name + ".png")), img)
        saved[f"norm_{m['idx']:05d}"] = img
    np.savez_compressed(Path(a.out) / "normalized.npz", **saved)

    print(f"\nНормализовано: {len(saved)}/{len(normed)}")
    print(f"Папка:  {norm_dir}/")
    print(f"Массив: {Path(a.out) / 'normalized.npz'}")


if __name__ == "__main__":
    main()
