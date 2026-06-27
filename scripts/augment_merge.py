#!/usr/bin/env python3
"""
Объединение датасета + флип-аугментация для дообучения (finetune).

База (--base, напр. datasets/v1) копируется как есть. Доп. размеченные кадры
(--extra-img/--extra-lbl, напр. OZON) делятся train/val по базовым кадрам
(без утечки флипов в val). К train extra применяются флипы h/v/hv
(картинка + пересчёт боксов). 4K-кадры даунскейлятся до --width (метки
нормализованы — не страдают). val/test остаются без флипов.

Пишет <out>/{images,labels}/{train,val,test}, data_v2.yaml (в корне),
<out>/dataset_manifest.json.

Пример:
  .venv/bin/python scripts/augment_merge.py --base datasets/v1 \
      --extra-img OZONVIDEOS/ft/images --extra-lbl OZONVIDEOS/ft/labels \
      --out datasets/v2 --val 0.2 --flips h,v,hv --width 1920 --clean
"""
from __future__ import annotations
import argparse, json, random, shutil
from collections import Counter
from datetime import datetime
from pathlib import Path
import cv2

CLASS_NAMES = {0: "qr", 1: "barcode_1d", 2: "datamatrix", 3: "pdf417", 4: "aztec"}
IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".PNG", ".JPG", ".JPEG")


def parse_args():
    p = argparse.ArgumentParser(description="merge base YOLO dataset + flip-augmented extra frames")
    p.add_argument("--base", default="datasets/v1")
    p.add_argument("--extra-img", default="OZONVIDEOS/ft/images")
    p.add_argument("--extra-lbl", default="OZONVIDEOS/ft/labels")
    p.add_argument("--out", default="datasets/v2")
    p.add_argument("--data-out", default="data_v2.yaml")
    p.add_argument("--val", type=float, default=0.2, help="доля extra в val")
    p.add_argument("--flips", default="h,v,hv")
    p.add_argument("--width", type=int, default=1920, help="даунскейл extra до ширины")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--clean", action="store_true")
    return p.parse_args()


def flip_lines(lines, mode):
    out = []
    for ln in lines:
        ps = ln.split()
        if len(ps) != 5:
            continue
        c, xc, yc, w, h = ps[0], float(ps[1]), float(ps[2]), float(ps[3]), float(ps[4])
        if mode in ("h", "hv"):
            xc = 1 - xc
        if mode in ("v", "hv"):
            yc = 1 - yc
        out.append(f"{c} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
    return out


def flip_img(img, mode):
    return {"h": cv2.flip(img, 1), "v": cv2.flip(img, 0), "hv": cv2.flip(img, -1)}[mode]


def downscale(img, width):
    h, w = img.shape[:2]
    if width and w > width:
        img = cv2.resize(img, (width, int(h * width / w)), interpolation=cv2.INTER_AREA)
    return img


def main():
    a = parse_args(); random.seed(a.seed)
    base, out = Path(a.base), Path(a.out)
    flips = [f for f in a.flips.split(",") if f]
    if a.clean and out.exists():
        shutil.rmtree(out)
    for sub in ("images", "labels"):
        for sp in ("train", "val", "test"):
            (out / sub / sp).mkdir(parents=True, exist_ok=True)

    detail = {"base": {}, "extra_val": 0, "extra_train_orig": 0, "flip_copies": 0}

    # 1) база — копируем как есть
    for sp in ("train", "val", "test"):
        bi = base / "images" / sp; n = 0
        if bi.is_dir():
            for ip in bi.iterdir():
                if ip.suffix not in IMG_EXTS:
                    continue
                shutil.copy2(ip, out / "images" / sp / ip.name)
                lp = base / "labels" / sp / f"{ip.stem}.txt"
                (out / "labels" / sp / f"{ip.stem}.txt").write_text(
                    lp.read_text() if lp.exists() else "")
                n += 1
        detail["base"][sp] = n

    # 2) extra — сплит по базовым кадрам
    exi = Path(a.extra_img)
    stems = sorted(p.stem for p in exi.iterdir() if p.suffix in IMG_EXTS)
    ext_path = {p.stem: p for p in exi.iterdir() if p.suffix in IMG_EXTS}
    random.shuffle(stems)
    nval = int(len(stems) * a.val)
    val_stems, train_stems = set(stems[:nval]), stems[nval:]

    def lbl_lines(stem):
        lp = Path(a.extra_lbl) / f"{stem}.txt"
        return lp.read_text().splitlines() if lp.exists() else []

    # val: только оригиналы (даунскейл)
    for stem in sorted(val_stems):
        img = downscale(cv2.imread(str(ext_path[stem])), a.width)
        cv2.imwrite(str(out / "images" / "val" / f"ozon_{stem}.jpg"), img,
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        (out / "labels" / "val" / f"ozon_{stem}.txt").write_text("\n".join(lbl_lines(stem)))
    detail["extra_val"] = len(val_stems)

    # train: оригинал + флипы (даунскейл)
    for stem in sorted(train_stems):
        img = downscale(cv2.imread(str(ext_path[stem])), a.width)
        lines = lbl_lines(stem)
        cv2.imwrite(str(out / "images" / "train" / f"ozon_{stem}.jpg"), img,
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        (out / "labels" / "train" / f"ozon_{stem}.txt").write_text("\n".join(lines))
        detail["extra_train_orig"] += 1
        for m in flips:
            cv2.imwrite(str(out / "images" / "train" / f"ozon_{stem}_{m}.jpg"),
                        flip_img(img, m), [cv2.IMWRITE_JPEG_QUALITY, 92])
            (out / "labels" / "train" / f"ozon_{stem}_{m}.txt").write_text(
                "\n".join(flip_lines(lines, m)))
            detail["flip_copies"] += 1

    # 3) data.yaml + manifest
    names_block = "\n".join(f"  {i}: {n}" for i, n in CLASS_NAMES.items())
    Path(a.data_out).write_text(
        f"# Датасет v2 = v1 + OZON (флип-аугментация). Запуск из корня проекта.\n"
        f"path: ./{out.as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\n"
        f"names:\n{names_block}\n")
    per = {sp: len(list((out / "images" / sp).glob("*"))) for sp in ("train", "val", "test")}
    box = Counter()
    for sp in ("train", "val", "test"):
        for tp in (out / "labels" / sp).glob("*.txt"):
            for ln in tp.read_text().splitlines():
                if ln.split():
                    box[CLASS_NAMES[int(ln.split()[0])]] += 1
    manifest = {"created": datetime.now().isoformat(timespec="seconds"), "base": a.base,
                "extra": a.extra_img, "seed": a.seed, "flips": flips, "width": a.width,
                "val_frac_extra": a.val, "images_per_split": per,
                "boxes_per_class": dict(box), "detail": detail, "classes": CLASS_NAMES}
    (out / "dataset_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))

    print("=== augment_merge готово ===")
    print(f"  out: {out}  | data.yaml: {a.data_out}")
    print(f"  изображений: train {per['train']} / val {per['val']} / test {per['test']}")
    print(f"  боксы по классам: {dict(box)}")
    print(f"  база: {detail['base']} | extra train(ориг) {detail['extra_train_orig']} "
          f"+ флипов {detail['flip_copies']} | extra val {detail['extra_val']}")


if __name__ == "__main__":
    main()
