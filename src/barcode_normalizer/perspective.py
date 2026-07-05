"""Геометрическая нормализация: поворот к вертикали + коррекция перспективы.

Новый подход (ориентация ведёт геометрию):
  1. Оценённый угол полос используем, чтобы повернуть кроп так, что полосы
     становятся ВЕРТИКАЛЬНЫМИ (rotate на angle-90).
  2. На повёрнутом изображении зона полос почти axis-aligned. Находим её
     четырёхугольник (может быть трапецией из-за перспективы).
  3. Гомография этого четырёхугольника в прямоугольник фиксированной
     высоты (апскейл встроен). Полосы остаются вертикальными.

Так мы не зависим от непредсказуемого порядка вершин minAreaRect.
"""
import cv2
import numpy as np

from .config import NormalizerConfig


def rotate_bound(img, angle_deg, border_value=None, flags=cv2.INTER_LANCZOS4):
    """Поворот вокруг центра с расширением холста (без обрезки)."""
    h, w = img.shape[:2]
    c = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(c, angle_deg, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw = int(h * sin + w * cos)
    nh = int(h * cos + w * sin)
    M[0, 2] += (nw - w) / 2.0
    M[1, 2] += (nh - h) / 2.0
    if border_value is None:
        border = cv2.BORDER_REPLICATE
        return cv2.warpAffine(img, M, (nw, nh), flags=flags,
                              borderMode=border), M
    return cv2.warpAffine(img, M, (nw, nh), flags=flags,
                          borderMode=cv2.BORDER_CONSTANT,
                          borderValue=border_value), M


def order_corners(pts: np.ndarray) -> np.ndarray:
    """Упорядочивает 4 точки: TL, TR, BR, BL (для axis-aligned-ish квадов)."""
    pts = np.array(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    d = (pts[:, 0] - pts[:, 1])
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmax(d)]
    bl = pts[np.argmin(d)]
    return np.array([tl, tr, br, bl], dtype=np.float32)

def _dist(a, b):
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))


def expand_quad_x(corners, frac, img_shape):
    """Расширяет квад по горизонтали на frac средней ширины (quiet zone).

    Двигаем левую пару углов против направления верхней/нижней сторон,
    правую — по нему; координаты зажимаются в границы изображения.
    """
    tl, tr, br, bl = order_corners(corners)
    top = tr - tl
    bot = br - bl
    n_top = float(np.linalg.norm(top)) + 1e-6
    n_bot = float(np.linalg.norm(bot)) + 1e-6
    width = (n_top + n_bot) / 2.0
    d_top = top / n_top * frac * width
    d_bot = bot / n_bot * frac * width
    q = np.array([tl - d_top, tr + d_top, br + d_bot, bl - d_bot],
                 dtype=np.float32)
    h, w = img_shape[:2]
    q[:, 0] = np.clip(q[:, 0], 0, w - 1)
    q[:, 1] = np.clip(q[:, 1], 0, h - 1)
    return q


def _valid_quad(corners, max_ratio=3.0):
    """Проверяет, что четырёхугольник не является вырожденным."""
    corners = order_corners(corners)
    tl, tr, br, bl = corners

    top = _dist(tl, tr)
    bottom = _dist(bl, br)
    left = _dist(tl, bl)
    right = _dist(tr, br)

    sides = [top, bottom, left, right]

    if min(sides) < 4:
        return False

    width_ratio = max(top, bottom) / max(min(top, bottom), 1e-6)
    height_ratio = max(left, right) / max(min(left, right), 1e-6)

    if width_ratio > max_ratio:
        return False

    if height_ratio > max_ratio:
        return False

    contour = corners.reshape(-1, 1, 2).astype(np.float32)

    if not cv2.isContourConvex(contour):
        return False

    if abs(cv2.contourArea(contour)) < 16:
        return False

    return True


def quad_from_mask(region_mask: np.ndarray, cfg=None):
    """Извлекает настоящий четырёхугольник из маски.

    Возвращает точки TL, TR, BR, BL или None.
    """
    if region_mask is None:
        return None

    mask = region_mask.astype(np.uint8)

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)

    if cv2.contourArea(contour) < 16:
        return None

    # Удаляем локальные вогнутости и шум.
    hull = cv2.convexHull(contour)

    perimeter = cv2.arcLength(hull, True)
    hull_area = abs(cv2.contourArea(hull))

    if perimeter <= 0 or hull_area <= 0:
        return None

    max_ratio = (
        cfg.max_perspective_ratio
        if cfg is not None
        else 3.0
    )

    best_quad = None
    best_error = float("inf")

    # Перебираем несколько степеней упрощения контура.
    for eps_ratio in (
        0.01,
        0.015,
        0.02,
        0.025,
        0.03,
        0.04,
        0.05,
        0.07,
        0.10
    ):
        approx = cv2.approxPolyDP(
            hull,
            eps_ratio * perimeter,
            True
        )

        if len(approx) != 4:
            continue

        if not cv2.isContourConvex(approx):
            continue

        quad = order_corners(approx.reshape(4, 2))

        if not _valid_quad(quad, max_ratio=max_ratio):
            continue

        quad_area = abs(
            cv2.contourArea(
                quad.reshape(-1, 1, 2)
            )
        )

        # Четырёхугольник не должен терять большую часть области.
        coverage = quad_area / hull_area

        if coverage < 0.65:
            continue

        error = abs(1.0 - coverage)

        if error < best_error:
            best_error = error
            best_quad = quad

    return best_quad

def bbox_from_mask(region_mask: np.ndarray):
    """Axis-aligned bounding box зоны полос -> 4 угла."""
    ys, xs = np.nonzero(region_mask)
    if len(xs) == 0:
        return None
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)


def local_bar_tilt(gray_region):
    """Локальный наклон полос (tan угла от вертикали) в полосе изображения.

    После общего поворота к вертикали остаточный наклон в разных
    частях говорит о перспективе (полосы сходятся).
    tan>0 — верх полос смещён вправо относительно низа.
    """
    g = gray_region.astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    # угол градиента; для вертикальных полос gx>>gy. Наклон = gy/gx полосы.
    w = np.abs(gx)
    tilt = np.where(np.abs(gx) > 1e-3, -gy / (gx + 1e-6), 0.0)
    # взвешенная медиана по сильным краям
    flat_t = tilt.ravel()
    flat_w = w.ravel()
    idx = flat_w > np.percentile(flat_w, 75)
    if not np.any(idx):
        return 0.0
    vals = flat_t[idx]
    return float(np.median(np.clip(vals, -3, 3)))


def estimate_shear(gray_rot_region):
    """Оценивает разницу наклона полос сверху и снизу зоны.

    Возвращает (tilt_top, tilt_bottom). Разница => перспектива.
    """
    h = gray_rot_region.shape[0]
    top = gray_rot_region[: h // 2]
    bot = gray_rot_region[h // 2:]
    return local_bar_tilt(top), local_bar_tilt(bot)


def build_target_size(corners: np.ndarray, cfg: NormalizerConfig):
    """Размер целевого прямоугольника. corners упорядочены TL,TR,BR,BL,
    полосы вертикальны => высота = вертикальная сторона (TL->BL)."""
    tl, tr, br, bl = corners
    w_top = _dist(tl, tr)
    w_bot = _dist(bl, br)
    h_left = _dist(tl, bl)
    h_right = _dist(tr, br)
    width = max(1.0, (w_top + w_bot) / 2.0)
    height = max(1.0, (h_left + h_right) / 2.0)

    out_h = cfg.target_height
    aspect = width / height
    out_w = int(round(out_h * aspect)) if cfg.keep_aspect else out_h
    out_w = max(1, min(out_w, cfg.max_width))
    return out_w, out_h


def warp_to_rect(img, corners, out_w, out_h, cfg):
    src = order_corners(corners)
    dst = np.array([[0, 0], [out_w - 1, 0],
                    [out_w - 1, out_h - 1], [0, out_h - 1]], dtype=np.float32)
    H = cv2.getPerspectiveTransform(src, dst)
    interp = cv2.INTER_LANCZOS4 if cfg.interpolation == "lanczos" else cv2.INTER_CUBIC
    return cv2.warpPerspective(img, H, (out_w, out_h), flags=interp,
                               borderMode=cv2.BORDER_REPLICATE)
