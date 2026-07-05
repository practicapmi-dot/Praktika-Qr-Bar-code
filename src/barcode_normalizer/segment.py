"""Сегментация зоны полос штрих-кода внутри кропа.

Bbox может захватывать фон и текст (цифры под кодом, надписи этикетки).
Нужно выделить именно прямоугольную область полос, чтобы:
  - оценка угла и границ не сбивалась о текст/фон;
  - можно было найти 4 угла кода для гомографии.

Идея: строим "карту штрихкодовости" = высокая когерентность структурного
тензора (единое направление градиента). Текст даёт разнонаправленные
градиенты (низкая когерентность), фон — слабые. Затем морфология и выбор
крупнейшего связного региона.
"""
import cv2
import numpy as np

from .config import NormalizerConfig


def barcode_score_map(coherence: np.ndarray, energy: np.ndarray) -> np.ndarray:
    """Комбинируем когерентность и нормированную энергию градиента."""
    e = energy.copy()
    e = e / (e.max() + 1e-6)
    score = coherence * np.sqrt(e)  # sqrt смягчает доминирование сильных краёв
    score = cv2.normalize(score, None, 0, 1, cv2.NORM_MINMAX)
    return score


def segment_barcode_region(score: np.ndarray, cfg: NormalizerConfig):
    """Возвращает (mask, rect) где rect = cv2.minAreaRect крупнейшего региона.

    mask — бинарная маска зоны полос. rect может быть None, если не найдено.
    """
    h, w = score.shape[:2]
    m = (score >= cfg.coherence_thresh).astype(np.uint8) * 255

    # Объединяем соседние вертикальные полосы преимущественно
    # по горизонтальному направлению.
    kx = max(3, int(w * cfg.morph_close_frac))
    ky = max(3, int(h * 0.03))

    if kx % 2 == 0:
        kx += 1

    if ky % 2 == 0:
        ky += 1

    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (kx, ky)
    )

    m = cv2.morphologyEx(
        m,
        cv2.MORPH_CLOSE,
        close_kernel
    )

    # Удаляем небольшие изолированные шумовые области.
    open_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (3, 3)
    )

    m = cv2.morphologyEx(
        m,
        cv2.MORPH_OPEN,
        open_kernel
    )

    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return m, None

    min_area = cfg.min_region_frac * h * w
    best_idx, best_area = -1, 0
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area and area > best_area:
            best_area = area
            best_idx = i
    if best_idx < 0:
        # fallback: крупнейший вообще
        areas = stats[1:, cv2.CC_STAT_AREA]
        best_idx = 1 + int(np.argmax(areas))

    region = (labels == best_idx).astype(np.uint8) * 255
    cnts, _ = cv2.findContours(region, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return region, None
    cnt = max(cnts, key=cv2.contourArea)
    rect = cv2.minAreaRect(cnt)  # ((cx,cy),(w,h),angle)
    return region, rect
