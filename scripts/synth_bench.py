#!/usr/bin/env python3
"""
Синтетический decode-бенчмарк нормализатора.

Генерирует идеальные Code128 (zxingcpp.write_barcode), портит их известными
искажениями (перспектива, поворот 0-180°, даунскейл до реального размера,
blur, шум) и меряет главную метрику — сколько кодов декодируется zxing-ом
ПОСЛЕ нормализации (и сколько без неё). Позволяет объективно сравнивать
версии/параметры нормализатора.

Пример:
  .venv/bin/python scripts/synth_bench.py --cases 120 --variants all
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import zxingcpp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from barcode_normalizer import NormalizerConfig, normalize_barcode  # noqa: E402

# Именованные варианты конфига для абляции.
VARIANTS = {
    "default": {},
    "no-sharpen": {"sharpen_amount": 0.0},
    "sharpen-1.0": {"sharpen_amount": 1.0},
    "quiet-0.08": {"quiet_zone_frac": 0.08},
    "no-2pass": {"second_pass": False},
    "no-perspective": {"correct_perspective": False},
}


def make_distorted(rng, text, angle, persp_px, out_w, blur_sigma):
    img = zxingcpp.write_barcode(zxingcpp.BarcodeFormat.Code128, text,
                                 width=400, height=160, quiet_zone=20)
    g = np.array(img, dtype=np.uint8)
    if g.ndim == 3:
        g = cv2.cvtColor(g, cv2.COLOR_RGB2GRAY)
    pad = 40
    canvas = np.full((g.shape[0] + 2 * pad, g.shape[1] + 2 * pad), 190, np.uint8)
    canvas[pad:-pad, pad:-pad] = g
    h, w = canvas.shape
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = src + rng.uniform(-persp_px, persp_px, (4, 2)).astype(np.float32)
    canvas = cv2.warpPerspective(canvas, cv2.getPerspectiveTransform(src, dst),
                                 (w, h), borderValue=190)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    M[0, 2] += (nw - w) / 2
    M[1, 2] += (nh - h) / 2
    canvas = cv2.warpAffine(canvas, M, (nw, nh), borderValue=190)
    scale = out_w / canvas.shape[1]
    canvas = cv2.resize(canvas, (out_w, max(8, int(canvas.shape[0] * scale))),
                        interpolation=cv2.INTER_AREA)
    if blur_sigma > 0:
        canvas = cv2.GaussianBlur(canvas, (0, 0), blur_sigma)
    canvas = np.clip(canvas.astype(np.float32) + rng.normal(0, 4, canvas.shape),
                     0, 255).astype(np.uint8)
    return cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)


def dec(img):
    try:
        for r in zxingcpp.read_barcodes(img):
            if r.text:
                return r.text
    except Exception:
        pass
    return None


def gen_cases(n, seed):
    rng = np.random.RandomState(seed)
    cases = []
    for i in range(n):
        text = f"PKG{1000 + i}"
        cases.append((text, make_distorted(
            rng, text, angle=rng.uniform(0, 180),
            persp_px=rng.uniform(0, 12), out_w=int(rng.uniform(120, 260)),
            blur_sigma=rng.uniform(0.4, 1.1))))
    return cases


def run_variant(tag, cases, cfg_kwargs):
    cfg = NormalizerConfig(**cfg_kwargs)
    ok = fail = 0
    for text, img in cases:
        h, w = img.shape[:2]
        try:
            n = normalize_barcode(img, (0, 0, w, h), cfg)
        except Exception:
            n = None
        if n is None:
            fail += 1
        elif dec(n) == text:
            ok += 1
    total = len(cases)
    print(f"{tag:<18} decode {ok}/{total} ({100 * ok / total:.1f}%)"
          + (f" | fail {fail}" if fail else ""))
    return ok


def main():
    p = argparse.ArgumentParser(description="Synthetic normalizer benchmark.")
    p.add_argument("--cases", type=int, default=60)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--variants", default="default",
                   help="имена вариантов через запятую, 'all' — все")
    a = p.parse_args()

    cases = gen_cases(a.cases, a.seed)
    raw_ok = sum(dec(img) == text for text, img in cases)
    print(f"кейсов: {len(cases)} | raw decode {raw_ok}/{len(cases)}"
          f" ({100 * raw_ok / len(cases):.1f}%)\n")

    names = list(VARIANTS) if a.variants == "all" else a.variants.split(",")
    for tag in names:
        run_variant(tag, cases, VARIANTS[tag.strip()])


if __name__ == "__main__":
    main()
