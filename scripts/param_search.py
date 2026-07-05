#!/usr/bin/env python3
"""
Random search оптимальных параметров NormalizerConfig.

Каждый кандидат оценивается decode-rate'ом на синтетическом бенче
(scripts/synth_bench.py). Лучшие проверяются на ВТОРОМ сиде (защита от
переобучения под конкретный набор искажений). Результаты пишутся в
runs/param_search/results.jsonl, лучший конфиг — в best_config.json.

Пример:
  .venv/bin/python scripts/param_search.py --n 64 --cases 100 --workers 6
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from synth_bench import gen_cases, dec  # noqa: E402
from barcode_normalizer import NormalizerConfig, normalize_barcode  # noqa: E402

SPACE = {
    "target_height": lambda r: int(r.choice([192, 256, 320, 384])),
    "sharpen_amount": lambda r: float(r.uniform(0.6, 1.6)),
    "sharpen_sigma": lambda r: float(r.uniform(0.8, 2.0)),
    "quiet_zone_frac": lambda r: float(r.uniform(0.02, 0.09)),
    "second_pass_min_deg": lambda r: float(r.uniform(1.0, 2.5)),
    "pad_ratio": lambda r: float(r.uniform(0.08, 0.18)),
    "coherence_thresh": lambda r: float(r.uniform(0.25, 0.45)),
    "morph_close_frac": lambda r: float(r.uniform(0.10, 0.22)),
    "clahe_clip": lambda r: float(r.uniform(1.5, 3.0)),
    "bg_kernel_frac": lambda r: float(r.uniform(0.35, 0.65)),
}

_CASES = None  # генерируется в каждом воркере (spawn: чистый процесс, без fork-дедлоков cv2)


def _init_worker(cases_n, seed):
    """Инициализация spawn-воркера: свои потоки cv2 и свой набор кейсов.

    Кейсы детерминированы сидом — у всех воркеров одинаковые.
    """
    import cv2
    cv2.setNumThreads(1)  # воркеры не должны драться за ядра
    global _CASES
    _CASES = gen_cases(cases_n, seed)


def sample_config(rng):
    return {k: fn(rng) for k, fn in SPACE.items()}


def score_config(job):
    idx, kw = job
    cfg = NormalizerConfig(**kw)
    ok = 0
    for text, img in _CASES:
        h, w = img.shape[:2]
        try:
            n = normalize_barcode(img, (0, 0, w, h), cfg)
        except Exception:
            n = None
        if n is not None and dec(n) == text:
            ok += 1
    return idx, ok


def main():
    p = argparse.ArgumentParser(description="Random search for NormalizerConfig.")
    p.add_argument("--n", type=int, default=64, help="число случайных конфигов")
    p.add_argument("--cases", type=int, default=100)
    p.add_argument("--seed", type=int, default=42, help="сид кейсов поиска")
    p.add_argument("--val-seed", type=int, default=123, help="сид валидации топа")
    p.add_argument("--top", type=int, default=5, help="сколько лучших валидировать")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--out", default="runs/param_search")
    a = p.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(7)
    candidates = [{}] + [sample_config(rng) for _ in range(a.n)]  # {} = текущие дефолты

    # spawn: чистые воркеры без унаследованных cv2/OpenMP-потоков (fork с ними
    # дедлочится). Результаты пишем инкрементально — прогресс виден, обрыв не
    # теряет посчитанное.
    ctx = get_context("spawn")
    scores = {}
    t0 = time.time()
    with ctx.Pool(a.workers, initializer=_init_worker,
                  initargs=(a.cases, a.seed)) as pool, \
            open(out / "results.jsonl", "w") as f:
        for done, (idx, s) in enumerate(
                pool.imap_unordered(score_config, list(enumerate(candidates))), 1):
            scores[idx] = s
            f.write(json.dumps({"decode": s, "cases": a.cases,
                                "cfg": candidates[idx]}, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{done}/{len(candidates)}] cfg#{idx} decode {s}/{a.cases}"
                  f" | {time.time() - t0:.0f}s", flush=True)

    results = sorted(((scores[i], kw) for i, kw in enumerate(candidates)),
                     key=lambda t: -t[0])
    print(f"\nпоиск: лучший {results[0][0]}/{a.cases}, дефолт {scores[0]}/{a.cases}")

    # Валидация топа на другом сиде — отсекаем переобучение под сид поиска.
    top = results[:a.top]
    with ctx.Pool(min(a.workers, len(top)), initializer=_init_worker,
                  initargs=(a.cases, a.val_seed)) as pool:
        val = dict(pool.imap_unordered(score_config,
                                       list(enumerate(kw for _, kw in top))))
    print("\nтоп на валидационном сиде:")
    best, best_total = None, -1
    for j, (s_search, kw) in enumerate(top):
        total = s_search + val[j]
        print(f"  search {s_search} + val {val[j]} = {total}  {kw or '(default)'}")
        if total > best_total:
            best_total, best = total, kw
    (out / "best_config.json").write_text(json.dumps(best, ensure_ascii=False, indent=2))
    print(f"\nлучший конфиг → {out / 'best_config.json'}")


if __name__ == "__main__":
    main()
