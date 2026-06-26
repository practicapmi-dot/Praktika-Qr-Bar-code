#!/usr/bin/env python3
"""
Подготовка датасета: Pascal VOC (xmin/ymin/xmax/ymax) -> YOLO (норм. xc/yc/w/h).

Делает: ремап классов, конвертацию боксов, опц. dedup точных дублей (по md5),
сплит train/val/test, раскладку <out>/images|labels/{train,val,test} и
dataset_manifest.json (версия + счётчики) для версионирования при пополнении.

Только stdlib — установка зависимостей не нужна.

Пример:
  python scripts/prepare_dataset.py \
      --src datasets/raw/barcode_qr --out datasets/v1 \
      --train 0.8 --val 0.1 --test 0.1 --seed 0 --clean
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

# Имя класса в VOC -> id в data.yaml (5 классов). Синонимы/опечатки — сюда же.
CLASS_MAP = {
    "qr": 0, "qrcode": 0, "qr_code": 0,
    "barcode": 1, "barcode_1d": 1, "1d": 1, "ean": 1, "ean13": 1, "code128": 1,
    "datamatrix": 2, "data_matrix": 2, "dmtx": 2,
    "pdf417": 3,
    "aztec": 4,
}
CLASS_NAMES = {0: "qr", 1: "barcode_1d", 2: "datamatrix", 3: "pdf417", 4: "aztec"}
IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG", ".BMP"]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="VOC -> YOLO + split + manifest.")
    p.add_argument("--src", default="datasets/raw/barcode_qr",
                   help="папка с *.xml и картинками")
    p.add_argument("--out", default="datasets/v1", help="выходной корень датасета")
    p.add_argument("--train", type=float, default=0.8)
    p.add_argument("--val", type=float, default=0.1)
    p.add_argument("--test", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--version-tag", dest="version_tag", default="v1")
    p.add_argument("--no-dedup", action="store_true", help="не выкидывать точные дубли")
    p.add_argument("--symlink", action="store_true", help="симлинки вместо копий")
    p.add_argument("--clean", action="store_true", help="очистить --out перед записью")
    return p.parse_args(argv)


def build_image_index(src: Path):
    """lower(name.ext) -> Path и lower(stem) -> [Path] для устойчивого матчинга."""
    by_name, by_stem = {}, defaultdict(list)
    for f in src.iterdir():
        if f.is_file() and f.suffix in IMG_EXTS:
            by_name[f.name.lower()] = f
            by_stem[f.stem.lower()].append(f)
    return by_name, by_stem


def resolve_image(xml_path: Path, filename, by_name, by_stem):
    if filename:
        hit = by_name.get(Path(filename).name.lower())
        if hit:
            return hit
    cands = by_stem.get(xml_path.stem.lower(), [])
    return cands[0] if cands else None


def voc_to_yolo(obj, w, h):
    name = (obj.findtext("name") or "").strip().lower()
    cid = CLASS_MAP.get(name)
    if cid is None:
        return None, name
    bb = obj.find("bndbox")
    if bb is None:
        return None, name
    xmin = float(bb.findtext("xmin")); ymin = float(bb.findtext("ymin"))
    xmax = float(bb.findtext("xmax")); ymax = float(bb.findtext("ymax"))
    xmin, xmax = sorted((xmin, xmax)); ymin, ymax = sorted((ymin, ymax))
    xmin = max(0.0, min(xmin, w)); xmax = max(0.0, min(xmax, w))
    ymin = max(0.0, min(ymin, h)); ymax = max(0.0, min(ymax, h))
    bw = xmax - xmin; bh = ymax - ymin
    if bw <= 1 or bh <= 1:
        return None, name  # вырожденный бокс
    xc = (xmin + xmax) / 2.0 / w
    yc = (ymin + ymax) / 2.0 / h
    return (cid, xc, yc, bw / w, bh / h), name


def md5(path: Path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def main(argv=None):
    a = parse_args(argv)
    src, out = Path(a.src), Path(a.out)
    if not src.is_dir():
        sys.exit(f"нет папки: {src}")
    ratios = (a.train, a.val, a.test)
    if abs(sum(ratios) - 1.0) > 1e-6:
        sys.exit(f"train+val+test должны давать 1.0 (сейчас {sum(ratios)})")

    by_name, by_stem = build_image_index(src)
    xmls = sorted(src.glob("*.xml"))
    if not xmls:
        sys.exit(f"в {src} нет *.xml")

    samples, seen = [], {}            # (img_path, [yolo_lines]); md5 -> name
    stats, unknown = Counter(), Counter()
    for xp in xmls:
        try:
            root = ET.parse(xp).getroot()
        except ET.ParseError as e:
            stats["xml_parse_error"] += 1
            print(f"  ! битый XML {xp.name}: {e}", file=sys.stderr)
            continue
        size = root.find("size")
        if size is None:
            stats["no_size"] += 1
            continue
        w = float(size.findtext("width")); h = float(size.findtext("height"))
        if w <= 0 or h <= 0:
            stats["bad_size"] += 1
            continue
        img = resolve_image(xp, root.findtext("filename"), by_name, by_stem)
        if img is None:
            stats["img_missing"] += 1
            print(f"  ! нет картинки для {xp.name}", file=sys.stderr)
            continue
        if not a.no_dedup:
            hsh = md5(img)
            if hsh in seen:
                stats["dup_dropped"] += 1
                continue
            seen[hsh] = img.name
        lines = []
        for obj in root.findall("object"):
            yolo, name = voc_to_yolo(obj, w, h)
            if yolo is None:
                if name not in CLASS_MAP:
                    unknown[name] += 1
                stats["box_skipped"] += 1
                continue
            cid, xc, yc, bw, bh = yolo
            lines.append(f"{cid} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
            stats[f"box_{CLASS_NAMES[cid]}"] += 1
        samples.append((img, lines))  # пустые lines => background-кадр (ок для YOLO)

    if not samples:
        sys.exit("нечего писать: 0 валидных образцов")

    random.seed(a.seed)
    random.shuffle(samples)
    n = len(samples)
    n_tr, n_va = int(n * ratios[0]), int(n * ratios[1])
    parts = {
        "train": samples[:n_tr],
        "val": samples[n_tr:n_tr + n_va],
        "test": samples[n_tr + n_va:],
    }

    if a.clean and out.exists():
        shutil.rmtree(out)
    for sub in ("images", "labels"):
        for sp in parts:
            (out / sub / sp).mkdir(parents=True, exist_ok=True)

    per_split, per_split_box = {}, {}
    for sp, items in parts.items():
        cc = Counter()
        for img, lines in items:
            dst = out / "images" / sp / img.name
            if a.symlink:
                if dst.exists() or dst.is_symlink():
                    dst.unlink()
                dst.symlink_to(img.resolve())
            else:
                shutil.copy2(img, dst)
            (out / "labels" / sp / (img.stem + ".txt")).write_text("\n".join(lines))
            for ln in lines:
                cc[CLASS_NAMES[int(ln.split()[0])]] += 1
        per_split[sp] = len(items)
        per_split_box[sp] = dict(cc)

    manifest = {
        "version_tag": a.version_tag,
        "created": datetime.now().isoformat(timespec="seconds"),
        "source": str(src),
        "seed": a.seed,
        "ratios": {"train": ratios[0], "val": ratios[1], "test": ratios[2]},
        "dedup": (not a.no_dedup),
        "classes": CLASS_NAMES,
        "totals": {"images": n, "xml_seen": len(xmls)},
        "per_split_images": per_split,
        "per_split_boxes": per_split_box,
        "diagnostics": dict(stats),
        "unknown_class_names": dict(unknown),
    }
    (out / "dataset_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2))

    print("=== prepare_dataset: готово ===")
    print(f"  out: {out}")
    print(f"  образцов: {n}  (train {per_split['train']} / "
          f"val {per_split['val']} / test {per_split['test']})")
    print(f"  боксы train: {per_split_box['train']}")
    print(f"  боксы val:   {per_split_box['val']}")
    print(f"  боксы test:  {per_split_box['test']}")
    if stats:
        print(f"  диагностика: {dict(stats)}")
    if unknown:
        print(f"  ! неизвестные классы (пропущены): {dict(unknown)}")
    print(f"  манифест: {out / 'dataset_manifest.json'}")


if __name__ == "__main__":
    main()
