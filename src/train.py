#!/usr/bin/env python3
"""
Гибкое обучение YOLO-детектора кодов (QR / 1D / DataMatrix / PDF417 / Aztec).

Три режима:
  fresh    — обучение с нуля от предобученных COCO-весов (yolo11n/s/m.pt).
  resume   — продолжение прерванного run с last.pt (+ состояние оптимизатора).
  finetune — дообучение от best.pt на (возможно дополненном) датасете; новый run,
             обычно с пониженным lr0.

Приоритет параметров: CLI-флаг > --config (YAML) > встроенные DEFAULTS.

Примеры запуска — в конце файла.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import yaml
except ImportError:  # нужен только для --config
    yaml = None


# Встроенные значения по умолчанию (можно переопределить через --config или CLI).
DEFAULTS = {
    "model": "yolo11n.pt",
    "data": "data.yaml",
    "project": "runs",
    "name": "qr_yolo_v1",
    "epochs": 100,
    "imgsz": 640,
    "batch": 16,        # -1 = авто-подбор по VRAM
    "lr0": 0.01,        # для fresh; для finetune обычно 0.001
    "lr0_finetune": 0.001,
    "patience": 50,
    "seed": 0,
    "workers": 8,
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Гибкое обучение YOLO (fresh / resume / finetune).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--mode", choices=["fresh", "resume", "finetune"], required=True,
                   help="режим обучения")
    p.add_argument("--config", default=None,
                   help="YAML с дефолтами (например configs/train.yaml)")
    # данные / веса
    p.add_argument("--data", default=None, help="путь к data.yaml")
    p.add_argument("--model", default=None, help="стартовые веса для fresh (yolo11n.pt)")
    p.add_argument("--weights", default=None,
                   help="best.pt для finetune (или last.pt для resume)")
    p.add_argument("--resume-path", dest="resume_path", default=None,
                   help="явный путь к last.pt для resume")
    # расположение run
    p.add_argument("--project", default=None,
                   help="корень runs (в Colab — путь на Google Drive)")
    p.add_argument("--name", default=None, help="имя run")
    # гиперпараметры
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--imgsz", type=int, default=None)
    p.add_argument("--batch", type=int, default=None, help="-1 = авто")
    p.add_argument("--lr0", type=float, default=None)
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--device", default=None, help="0 / cpu / 0,1 (по умолчанию авто)")
    return p.parse_args(argv)


def load_config(path):
    if not path:
        return {}
    if yaml is None:
        sys.exit("PyYAML не установлен — нужен для --config (pip install pyyaml).")
    with open(path) as f:
        return yaml.safe_load(f) or {}


def resolve(cli, cfg, key, fallback=None):
    """CLI > config > DEFAULTS > fallback."""
    val = getattr(cli, key, None)
    if val is not None:
        return val
    if key in cfg and cfg[key] is not None:
        return cfg[key]
    if key in DEFAULTS:
        return DEFAULTS[key]
    return fallback


def auto_device(explicit):
    if explicit is not None:
        return explicit
    try:
        import torch
        if torch.cuda.is_available():
            return 0
    except Exception:
        pass
    return "cpu"


def find_latest_last_pt(project, name=None):
    """Найти last.pt: сначала project/name/weights, иначе самый свежий по проекту."""
    project = Path(project)
    if name:
        cand = project / name / "weights" / "last.pt"
        if cand.exists():
            return cand
    if not project.exists():
        return None
    found = sorted(project.glob("*/weights/last.pt"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    return found[0] if found else None


def _common_train_kwargs(cli, cfg, *, name, lr0):
    return dict(
        data=resolve(cli, cfg, "data"),
        epochs=resolve(cli, cfg, "epochs"),
        imgsz=resolve(cli, cfg, "imgsz"),
        batch=resolve(cli, cfg, "batch"),
        lr0=lr0,
        patience=resolve(cli, cfg, "patience"),
        seed=resolve(cli, cfg, "seed"),
        workers=resolve(cli, cfg, "workers"),
        project=resolve(cli, cfg, "project"),
        name=name,
        device=auto_device(cli.device),
        exist_ok=False,
    )


def train_fresh(cli, cfg):
    from ultralytics import YOLO
    weights = resolve(cli, cfg, "model")
    print(f"[fresh] стартовые веса: {weights}")
    model = YOLO(weights)
    model.train(**_common_train_kwargs(
        cli, cfg, name=resolve(cli, cfg, "name"), lr0=resolve(cli, cfg, "lr0")))
    return model


def train_resume(cli, cfg):
    from ultralytics import YOLO
    last_pt = cli.resume_path or cli.weights
    if last_pt:
        last_pt = Path(last_pt)
        if not last_pt.exists():
            sys.exit(f"[resume] не найден файл: {last_pt}")
    else:
        last_pt = find_latest_last_pt(resolve(cli, cfg, "project"),
                                      cli.name or cfg.get("name"))
        if not last_pt:
            sys.exit("[resume] last.pt не найден под --project. "
                     "Укажи --resume-path /путь/last.pt")
    print(f"[resume] продолжаю с: {last_pt}")
    model = YOLO(str(last_pt))
    # resume=True сам читает args.yaml из папки run и продолжает с того же места
    model.train(resume=True)
    return model


def train_finetune(cli, cfg):
    from ultralytics import YOLO
    weights = cli.weights or resolve(cli, cfg, "weights", None)
    if not weights:  # попробуем best.pt из последнего run
        last = find_latest_last_pt(resolve(cli, cfg, "project"),
                                   cli.name or cfg.get("name"))
        if last and (last.parent / "best.pt").exists():
            weights = str(last.parent / "best.pt")
    if not weights or not Path(weights).exists():
        sys.exit("[finetune] нужен --weights /путь/best.pt от предыдущего обучения.")
    ft_lr = cli.lr0 if cli.lr0 is not None else cfg.get("lr0_finetune",
                                                        DEFAULTS["lr0_finetune"])
    name = resolve(cli, cfg, "name")
    if name == DEFAULTS["name"]:  # не затираем исходный run по умолчанию
        name += "_ft"
    print(f"[finetune] от весов: {weights} | lr0={ft_lr} | name={name}")
    model = YOLO(weights)
    model.train(**_common_train_kwargs(cli, cfg, name=name, lr0=ft_lr))
    return model


def main(argv=None):
    cli = parse_args(argv)
    cfg = load_config(cli.config)
    print(f"=== train.py | mode={cli.mode} | device={auto_device(cli.device)} ===")
    handler = {"fresh": train_fresh, "resume": train_resume,
               "finetune": train_finetune}[cli.mode]
    model = handler(cli, cfg)
    save_dir = getattr(getattr(model, "trainer", None), "save_dir", None)
    if save_dir:
        best = Path(save_dir) / "weights" / "best.pt"
        print(f"[done] run: {save_dir}")
        print(f"[done] лучшие веса: {best}")
    else:
        print("[done] обучение завершено.")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# ПРИМЕРЫ ЗАПУСКА (конкретные флаги):
#
# 1) Обучение с нуля (fresh):
#    python src/train.py --mode fresh \
#        --data data.yaml --model yolo11n.pt \
#        --epochs 100 --imgsz 640 --batch 16 --device 0 \
#        --project runs --name qr_yolo_v1
#
# 2) Продолжить прерванное обучение (resume) — авто-поиск last.pt по --project/--name:
#    python src/train.py --mode resume --project runs --name qr_yolo_v1
#    # либо явно:
#    python src/train.py --mode resume --resume-path runs/qr_yolo_v1/weights/last.pt
#
# 3) Дообучение на дополненном датасете (finetune) — новый run, пониженный lr0:
#    python src/train.py --mode finetune \
#        --data data.yaml --weights runs/qr_yolo_v1/weights/best.pt \
#        --epochs 50 --lr0 0.001 --device 0 \
#        --project runs --name qr_yolo_v2_ft
#
# 4) С конфигом (configs/train.yaml как дефолты, CLI переопределяет):
#    python src/train.py --mode fresh --config configs/train.yaml --device 0
# ---------------------------------------------------------------------------
