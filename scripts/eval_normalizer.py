#!/usr/bin/env python3
"""
Оценка качества нормализатора на реальных кропах пайплайна.

Берёт crops_meta.json от track_crops/pipeline_video, повторно вырезает СЫРЫЕ
кропы из исходных видео (по video/frame/xyxy, без YOLO — быстро) и сравнивает
варианты обработки тремя метриками:

  decode      — % кропов, которые декодирует zxing-cpp (главная метрика);
  verticality — насколько полосы вертикальны: var(профиль по столбцам) /
                (var по столбцам + var по строкам), 1.0 = идеальные верт. полосы;
  clean_bars  — % кропов, где число ч/б переходов стабильно от строки к строке
                (std/mean < 0.35) — признак ровной, не перекошенной сетки полос.

Пример:
  .venv/bin/python scripts/eval_normalizer.py \
      --meta runs/pipeline/crops_meta.json --videos OZONVIDEOS
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from track_crops import extract_crop  # noqa: E402
from barcode_normalizer import NormalizerConfig, normalize_barcode  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate barcode normalizer quality.")
    p.add_argument("--meta", default="runs/pipeline/crops_meta.json")
    p.add_argument("--videos", default="OZONVIDEOS", help="папка с исходными видео")
    p.add_argument("--cache", default="/tmp/eval_raws.npz", help="кэш сырых кропов")
    p.add_argument("--save-norm", default="", help="куда сохранить нормализованные PNG (опц.)")
    return p.parse_args()


def load_raws(meta, videos_dir, cache_path):
    """Сырые кропы по мете (кэшируются в npz, чтобы не перечитывать видео)."""
    cache = Path(cache_path)
    if cache.exists():
        data = np.load(cache)
        if len(data.files) == len(meta):
            return [data[f"raw_{i:05d}"] for i in range(len(meta))]
    by_video = defaultdict(list)
    for m in meta:
        by_video[m["video"]].append(m)
    raws = {}
    for vname, items in by_video.items():
        cap = cv2.VideoCapture(str(Path(videos_dir) / vname))
        for m in sorted(items, key=lambda m: m["frame"]):
            cap.set(cv2.CAP_PROP_POS_FRAMES, m["frame"])
            ok, frame = cap.read()
            if not ok:
                raise SystemExit(f"не читается кадр {m['frame']} из {vname}")
            raws[m["idx"]] = extract_crop(frame, m["xyxy"])
        cap.release()
    out = [raws[i] for i in range(len(meta))]
    np.savez_compressed(cache, **{f"raw_{i:05d}": c for i, c in enumerate(out)})
    return out


def decode_ok(img):
    import zxingcpp
    try:
        for r in zxingcpp.read_barcodes(img):
            if r.text:
                return True
    except Exception:
        pass
    return False


def bar_metrics(img):
    """(verticality 0..1, clean_bars bool) для одного изображения."""
    g = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gf = g.astype(np.float32)
    vc = float(np.var(gf.mean(axis=0)))
    vr = float(np.var(gf.mean(axis=1)))
    vert = vc / (vc + vr + 1e-6)
    _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    trans = np.count_nonzero(np.diff(bw.astype(np.int16), axis=1), axis=1)
    trans = trans[trans > 0]
    clean = (len(trans) > 0 and np.median(trans) >= 4
             and float(np.std(trans)) / (float(np.mean(trans)) + 1e-6) < 0.35)
    return vert, clean


def evaluate(tag, images):
    n = len(images)
    dec = sum(decode_ok(im) for im in images if im is not None)
    verts, cleans = [], 0
    skipped = sum(1 for im in images if im is None)
    for im in images:
        if im is None:
            continue
        v, c = bar_metrics(im)
        verts.append(v)
        cleans += bool(c)
    print(f"{tag:<22} decode {dec}/{n} ({100*dec/max(1,n):.1f}%)"
          f" | verticality {np.mean(verts):.3f}"
          f" | clean_bars {cleans}/{n} ({100*cleans/max(1,n):.1f}%)"
          + (f" | fail {skipped}" if skipped else ""))
    return dec, float(np.mean(verts)), cleans


def main():
    a = parse_args()
    meta = json.load(open(a.meta))
    meta = [m for m in meta if m["cls"] == "barcode_1d"]
    raws = load_raws(meta, a.videos, a.cache)
    print(f"кропов (barcode_1d): {len(raws)}\n")

    evaluate("raw", raws)

    cfg = NormalizerConfig()
    norm = []
    for r in raws:
        h, w = r.shape[:2]
        try:
            norm.append(normalize_barcode(r, (0, 0, w, h), cfg))
        except Exception:
            norm.append(None)
    evaluate("normalized", norm)

    if a.save_norm:
        d = Path(a.save_norm)
        d.mkdir(parents=True, exist_ok=True)
        for i, im in enumerate(norm):
            if im is not None:
                cv2.imwrite(str(d / f"{i:05d}.png"), im)
        print(f"\nнормализованные PNG: {d}/")


if __name__ == "__main__":
    main()
