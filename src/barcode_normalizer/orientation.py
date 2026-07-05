"""Оценка ориентации полос 1D штрих-кода.

Соглашение об угле: `angle` — направление ВДОЛЬ полос в градусах,
0° = горизонталь (полосы идут по оси X, т.е. лежачие),
90° = вертикаль (полосы стоячие). Цель нормализации — привести к 90°.

Основной метод — проекционный (Radon-подобный) поиск:
  для набора углов проецируем изображение на ось, перпендикулярную
  предполагаемому направлению полос, и измеряем "резкость" профиля.
  Когда направление проецирования совпадает с нормалью к полосам,
  профиль имеет максимальную дисперсию (чёткие пики светлых/тёмных).

Плюс тензор структуры — для карт когерентности/энергии (сегментация)
и как грубая начальная оценка.
"""
import cv2
import numpy as np

from .config import NormalizerConfig


def structure_tensor(gray: np.ndarray, win: int):
    """Карты для сегментации: orientation, coherence, energy."""
    g = gray.astype(np.float32) / 255.0
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    jxx, jyy, jxy = gx * gx, gy * gy, gx * gy
    if win % 2 == 0:
        win += 1
    ksize = (win, win)
    jxx = cv2.GaussianBlur(jxx, ksize, 0)
    jyy = cv2.GaussianBlur(jyy, ksize, 0)
    jxy = cv2.GaussianBlur(jxy, ksize, 0)
    tmp = np.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2)
    lam1 = 0.5 * (jxx + jyy + tmp)
    lam2 = 0.5 * (jxx + jyy - tmp)
    coherence = np.clip((lam1 - lam2) / (lam1 + lam2 + 1e-6), 0, 1)
    energy = lam1 + lam2
    return coherence, energy


def _rotate_expand(img, rot_deg):
    """Поворот с расширением холста, фон = 0 (для маскированных профилей)."""
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), rot_deg, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw = int(h * sin + w * cos)
    nh = int(h * cos + w * sin)
    M[0, 2] += (nw - w) / 2.0
    M[1, 2] += (nh - h) / 2.0
    return cv2.warpAffine(img, M, (nw, nh), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def _cross_profile(gray_f, mask_f, angle_along_bars_deg):
    """Взвешенный 1D-профиль поперёк полос при данном угле.

    Маска отсекает пиксели холста, добавленные поворотом: без неё
    BORDER_REFLECT создавал зеркальные фантомные полосы, а нули фона
    портили средние. Берём центральную половину строк (края/фон мешают)
    и только столбцы с достаточным покрытием реальными пикселями.
    """
    rot = angle_along_bars_deg - 90.0
    r = _rotate_expand(gray_f, rot)
    m = _rotate_expand(mask_f, rot)
    hh = r.shape[0]
    if hh >= 8:
        r = r[hh // 4: 3 * hh // 4]
        m = m[hh // 4: 3 * hh // 4]
    wsum = m.sum(axis=0)
    good = wsum > 0.5 * (wsum.max() + 1e-6)
    if int(good.sum()) < 8:
        return None
    prof = (r * m).sum(axis=0)[good] / wsum[good]
    return prof


def _projection_sharpness(gray_f, mask_f, angle_along_bars_deg):
    """Резкость профиля при проекции вдоль полос.

    Когда угол угадан, усреднение вдоль полос даёт максимально резкую
    «пилу». Меряем СРЕДНЮЮ энергию производной профиля: среднее (а не
    сумма) убирает систематическое смещение — длина повёрнутого холста
    зависит от угла, и сумма завышала счёт диагональных углов.
    """
    prof = _cross_profile(gray_f, mask_f, angle_along_bars_deg)
    if prof is None:
        return 0.0
    d = np.diff(prof)
    return float(np.mean(d * d))


def _vertical_periodicity(gray_f, mask_f, angle_along_bars_deg):
    """Пиковость спектра профиля: у настоящих полос выраженный период."""
    prof = _cross_profile(gray_f, mask_f, angle_along_bars_deg)
    if prof is None or len(prof) < 8:
        return 0.0
    prof = prof - prof.mean()
    spec = np.abs(np.fft.rfft(prof * np.hanning(len(prof))))
    if len(spec) < 4:
        return 0.0
    band = spec[2:]
    peak = float(np.max(band))
    mean = float(np.mean(band)) + 1e-6
    return peak * (peak / mean)


def estimate_angle_projection(gray, coarse_step=2.0, fine_step=0.5, max_side=256):
    """Угол ВДОЛЬ полос: перебор + параболическое уточнение + анти-90°.

    Изображение даунскейлится до max_side (глобальному профилю мелкие
    детали не нужны, а поворотов много), затем coarse-перебор 0..180,
    fine-перебор вокруг лучшего и параболическая интерполяция по трём
    точкам — суб-шаговая точность без лишних поворотов.
    """
    g = gray.astype(np.float32) / 255.0
    h, w = g.shape[:2]
    s = max_side / max(h, w)
    if s < 1.0:
        g = cv2.resize(g, (max(8, int(w * s)), max(8, int(h * s))),
                       interpolation=cv2.INTER_AREA)
    mask = np.ones_like(g, dtype=np.float32)

    best_a, best_s = 0.0, -1.0
    for a in np.arange(0, 180, coarse_step):
        sc = _projection_sharpness(g, mask, a)
        if sc > best_s:
            best_s, best_a = sc, float(a)
    for a in np.arange(best_a - coarse_step, best_a + coarse_step + 1e-6, fine_step):
        sc = _projection_sharpness(g, mask, a % 180.0)
        if sc > best_s:
            best_s, best_a = sc, float(a % 180.0)

    # Параболическая интерполяция по (best-δ, best, best+δ).
    s_m = _projection_sharpness(g, mask, (best_a - fine_step) % 180.0)
    s_p = _projection_sharpness(g, mask, (best_a + fine_step) % 180.0)
    denom = s_m - 2.0 * best_s + s_p
    if abs(denom) > 1e-9:
        delta = 0.5 * (s_m - s_p) / denom
        if abs(delta) <= 1.0:
            best_a = (best_a + delta * fine_step) % 180.0

    # Разрешение 90°-неоднозначности: где периодичность сильнее.
    cand = [best_a, (best_a - 90.0) % 180.0]
    scores = [_vertical_periodicity(g, mask, c) for c in cand]
    best = cand[int(np.argmax(scores))]
    return float(best % 180.0), float(best_s)


def estimate_orientation(gray_clean: np.ndarray, cfg: NormalizerConfig):
    """Итог: angle (вдоль полос, 90=вертикаль) + карты для сегментации."""
    coherence, energy = structure_tensor(gray_clean, cfg.coherence_win)
    angle, sharp = estimate_angle_projection(gray_clean)
    return {
        "angle": float(angle),
        "coherence_map": coherence,
        "energy_map": energy,
        "confidence": float(sharp),
        "method": "projection",
    }
