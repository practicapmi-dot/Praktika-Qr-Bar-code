"""Главный конвейер нормализации 1D штрих-кодов (ориентация ведёт геометрию).

Порядок:
  1. Кроп по bbox (+padding).
  2. Предобработка (grayscale, вычитание фона, CLAHE, denoise).
  3. Оценка угла полос (structure tensor + FFT). angle = направление ВДОЛЬ
     полос (90° = вертикаль).
  4. Поворот кропа на (angle - 90): полосы становятся вертикальными.
  5. Сегментация зоны полос на повёрнутом изображении.
  6. Поиск четырёхугольника зоны (трапеция => перспектива) и гомография
     в прямоугольник фиксированной высоты (апскейл встроен).
  7. Финальная проверка вертикальности + формат выхода.

API:
    normalize_barcode(image, bbox, cfg=None, debug=False)
    batch_normalize(image, bboxes, cfg=None)
"""
import cv2
import numpy as np

from .config import NormalizerConfig
from . import preprocess as pp
from . import orientation as ori
from . import segment as seg
from . import perspective as persp


def _crop_with_padding(image, bbox, cfg):
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox
    x1, x2 = sorted((int(x1), int(x2)))
    y1, y2 = sorted((int(y1), int(y2)))
    bw, bh = x2 - x1, y2 - y1
    px = int(round(bw * cfg.pad_ratio))
    py = int(round(bh * cfg.pad_ratio))
    nx1, ny1 = max(0, x1 - px), max(0, y1 - py)
    nx2, ny2 = min(w, x2 + px), min(h, y2 + py)
    return image[ny1:ny2, nx1:nx2].copy(), (nx1, ny1, nx2, ny2)


def _ensure_bars_vertical(warped):
    """Страховка: если вдруг полосы вышли горизонтальными — повернуть на 90°.

    У вертикальных полос дисперсия профиля по столбцам >> по строкам.
    """
    g = warped if warped.ndim == 2 else cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    g = g.astype(np.float32)
    var_cols = float(np.var(g.mean(axis=0)))  # вертикальные полосы
    var_rows = float(np.var(g.mean(axis=1)))  # горизонтальные полосы
    if var_rows > var_cols:
        warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
    return warped


def _remove_residual_tilt(warped, cfg):
    """Точный доповорот выпрямленного кода на остаточный угол.

    После гомографии полосы могут остаться наклонёнными на 1-3°: квад
    строился по маске, а не по самим полосам. Меряем угол повторно на
    результате (он маленький и чистый — оценка точная), поворачиваем и
    срезаем клинья REPLICATE-границ, затем возвращаем целевую высоту.
    """
    g = warped if warped.ndim == 2 else cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    angle, _ = ori.estimate_angle_projection(g)
    resid = angle - 90.0
    if abs(resid) < cfg.second_pass_min_deg or abs(resid) > 25.0:
        return warped
    h0, w0 = warped.shape[:2]
    rot, _ = persp.rotate_bound(warped, resid, flags=persp.interp_flag(cfg))
    rad = np.deg2rad(abs(resid))
    mx = int(np.ceil(h0 * np.sin(rad)))
    my = int(np.ceil(w0 * np.sin(rad)))
    nh, nw = rot.shape[:2]
    if nw - 2 * mx >= 16 and nh - 2 * my >= 16:
        rot = rot[my:nh - my, mx:nw - mx]
    h1, w1 = rot.shape[:2]
    scale = cfg.target_height / float(h1)
    # Масштаб близок к 1 → ресайз только повторно замылит полосы; не трогаем.
    if 0.9 <= scale <= 1.1:
        return rot
    out_w = max(1, min(int(round(w1 * scale)), cfg.max_width))
    return cv2.resize(rot, (out_w, cfg.target_height),
                      interpolation=persp.interp_flag(cfg))


def normalize_barcode(image, bbox, cfg: NormalizerConfig = None, debug=False):
    if cfg is None:
        cfg = NormalizerConfig()
    dbg = {} if debug else None

    crop, used_box = _crop_with_padding(image, bbox, cfg)
    if crop.size == 0:
        return (None, dbg) if debug else None
    if debug:
        dbg["crop"] = crop.copy()

    # 1. Предобработка.
    gray_clean, pp_steps = pp.preprocess(crop, cfg)
    if debug:
        dbg["preprocess"] = pp_steps

    # 2. Ориентация полос.
    ori_res = ori.estimate_orientation(gray_clean, cfg)
    angle = ori_res["angle"]  # направление вдоль полос; 90 = вертикаль
    if debug:
        dbg["orientation"] = ori_res

    
    # 3. Поворот к вертикали: полосы -> вертикальные.
    rot_angle = angle - 90.0
    crop_rot, M = persp.rotate_bound(crop, rot_angle, flags=persp.interp_flag(cfg))
    gray_rot, _ = persp.rotate_bound(gray_clean, rot_angle, flags=persp.interp_flag(cfg))
    if debug:
        dbg["crop_rot"] = crop_rot.copy()
        dbg["rot_angle"] = rot_angle
    
    
    # 4. Сегментация зоны полос на повёрнутом изображении.
    ori_rot = ori.estimate_orientation(gray_rot, cfg)
    score = seg.barcode_score_map(ori_rot["coherence_map"],
                                  ori_rot["energy_map"])
    region_mask, rect = seg.segment_barcode_region(score, cfg)
    if debug:
        dbg["score"] = score
        dbg["region_mask"] = region_mask

    # 5. Получение четырёхугольника области штрихкода.
    corners = None
    quad_mode = None

    if region_mask is not None and cv2.countNonZero(region_mask) > 0:
        if cfg.correct_perspective:
            # Настоящий четырёхугольник по контуру маски.
            corners = persp.quad_from_mask(region_mask, cfg)
            if corners is not None:
                quad_mode = "contour_quad"

        # Если четырёхугольник получить не удалось,
        # используем повёрнутый прямоугольник как fallback.
        if corners is None and rect is not None:
            corners = persp.order_corners(cv2.boxPoints(rect))
            quad_mode = "min_area_rect"

        # Следующий fallback — обычный bbox.
        if corners is None:
            corners = persp.bbox_from_mask(region_mask)
            quad_mode = "axis_aligned_bbox"

    # Последний fallback — весь повёрнутый кроп.
    if corners is None:
        h, w = crop_rot.shape[:2]
        corners = np.array([
            [0, 0],
            [w - 1, 0],
            [w - 1, h - 1],
            [0, h - 1]
        ], dtype=np.float32)
        quad_mode = "full_crop"

    # Quiet zone: расширяем квад по горизонтали, чтобы не срезать белые
    # поля по бокам кода — без них 1D-декодеры не находят старт/стоп.
    if getattr(cfg, "quiet_zone_frac", 0) > 0:
        corners = persp.expand_quad_x(corners, cfg.quiet_zone_frac,
                                      crop_rot.shape)

    if debug:
        dbg["corners"] = corners.copy()
        dbg["quad_mode"] = quad_mode

    # 6. Гомография зоны -> прямоугольник (апскейл встроен).
    out_w, out_h = persp.build_target_size(corners, cfg)
    warped = persp.warp_to_rect(crop_rot, corners, out_w, out_h, cfg)

    if cfg.orient_bars_vertical:
        warped = _ensure_bars_vertical(warped)

    # 7. Второй проход: гомография могла оставить остаточный наклон —
    # меряем угол уже на выпрямленном изображении и снимаем его точным
    # доповоротом (с обрезкой клиньев по краям).
    if getattr(cfg, "second_pass", False):
        warped = _remove_residual_tilt(warped, cfg)

    # 8. Лёгкий unsharp: подчёркивает переходы полос после Lanczos-апскейла.
    if getattr(cfg, "sharpen_amount", 0) > 0:
        blur = cv2.GaussianBlur(warped, (0, 0), cfg.sharpen_sigma)
        warped = cv2.addWeighted(warped, 1.0 + cfg.sharpen_amount,
                                 blur, -cfg.sharpen_amount, 0)

    # 9. Формат выхода.
    if cfg.output_grayscale and warped.ndim == 3:
        warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

    if debug:
        dbg["result"] = warped.copy()
        return warped, dbg
    return warped


def batch_normalize(image, bboxes, cfg: NormalizerConfig = None):
    if cfg is None:
        cfg = NormalizerConfig()
    out = []
    for bb in bboxes:
        try:
            out.append(normalize_barcode(image, bb, cfg, debug=False))
        except Exception as e:
            print("Ошибка при итерации:", len(out))
            print(e)
            out.append(None)
    return out
