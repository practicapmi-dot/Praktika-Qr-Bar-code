"""Визуализация промежуточных шагов конвейера для отладки."""
import cv2
import numpy as np


def _to_bgr(img):
    if img is None:
        return None
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def _label(img, text):
    img = img.copy()
    bar = np.full((22, img.shape[1], 3), 40, np.uint8)
    cv2.putText(bar, text, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def _upscale(img, target=180):
    h, w = img.shape[:2]
    s = max(1.0, target / max(h, w))
    return cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST)


def render_debug(dbg, save_path=None):
    """Собирает панель из ключевых шагов. Возвращает изображение (BGR)."""
    panels = []

    crop = dbg.get("crop")
    if crop is not None:
        panels.append(_label(_upscale(_to_bgr(crop)), "1. crop"))

    pre = dbg.get("preprocess", {})
    if "clahe" in pre:
        panels.append(_label(_upscale(_to_bgr(pre["clahe"])), "2. clahe"))
    elif "denoised" in pre:
        panels.append(_label(_upscale(_to_bgr(pre["denoised"])), "2. pre"))

    crop_rot = dbg.get("crop_rot")
    if crop_rot is not None:
        ra = dbg.get("rot_angle", 0.0)
        panels.append(_label(_upscale(_to_bgr(crop_rot)),
                             "2b. rotated %.0f" % ra))

    score = dbg.get("score")
    if score is not None:
        s = (score * 255).astype(np.uint8)
        s = cv2.applyColorMap(s, cv2.COLORMAP_JET)
        panels.append(_label(_upscale(s), "3. score"))

    region = dbg.get("region_mask")
    if region is not None:
        panels.append(_label(_upscale(_to_bgr(region)), "4. region"))

    # rotated crop + четырёхугольник зоны.
    if crop_rot is not None and dbg.get("corners") is not None:
        vis = _to_bgr(crop_rot).copy()
        c = dbg["corners"].astype(int)
        cv2.polylines(vis, [c.reshape(-1, 1, 2)], True, (0, 0, 255), 1)
        panels.append(_label(_upscale(vis), "5. quad on rotated"))

    res = dbg.get("result")
    if res is not None:
        r = _to_bgr(res)
        rr = cv2.resize(r, (int(r.shape[1] * 180 / r.shape[0]), 180),
                        interpolation=cv2.INTER_NEAREST)
        panels.append(_label(rr, "6. result (bars vertical)"))

    # выравниваем по высоте и склеиваем горизонтально
    if not panels:
        return None
    maxh = max(p.shape[0] for p in panels)
    fixed = []
    for p in panels:
        if p.shape[0] < maxh:
            pad = np.full((maxh - p.shape[0], p.shape[1], 3), 40, np.uint8)
            p = np.vstack([p, pad])
        fixed.append(p)
        fixed.append(np.full((maxh, 6, 3), 90, np.uint8))
    out = np.hstack(fixed[:-1])
    if save_path:
        cv2.imwrite(save_path, out)
    return out
