"""Предобработка кропа штрих-кода.

Задачи:
  - перевод в grayscale;
  - выравнивание неравномерного освещения (вычитание фона);
  - локальное повышение контраста (CLAHE);
  - мягкое шумоподавление с сохранением краёв полос.

Бинаризацию тут НЕ делаем: при очень низком разрешении жёсткий порог
убивает тонкие полосы. Для 1D-декодера важен плавный яркостный профиль.
"""
import cv2
import numpy as np

from .config import NormalizerConfig


def to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def subtract_background(gray: np.ndarray, cfg: NormalizerConfig) -> np.ndarray:
    """Убирает плавную неравномерность освещения / крупные блики.

    Оцениваем фон морфологическим закрытием большим ядром и вычитаем.
    """
    h, w = gray.shape[:2]
    k = max(3, int(min(h, w) * cfg.bg_kernel_frac))
    if k % 2 == 0:
        k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    # Нормализуем: gray / bg, чтобы выровнять освещение мультипликативно.
    bg_f = bg.astype(np.float32) + 1e-3
    norm = gray.astype(np.float32) / bg_f
    norm = np.clip(norm, 0, 1.0)
    norm = (norm * 255).astype(np.uint8)
    return norm


def apply_clahe(gray: np.ndarray, cfg: NormalizerConfig) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=cfg.clahe_clip,
                            tileGridSize=(cfg.clahe_grid, cfg.clahe_grid))
    return clahe.apply(gray)


def denoise(gray: np.ndarray, cfg: NormalizerConfig) -> np.ndarray:
    return cv2.bilateralFilter(gray, cfg.bilateral_d,
                               cfg.bilateral_sigma_color,
                               cfg.bilateral_sigma_space)


def preprocess(img: np.ndarray, cfg: NormalizerConfig):
    """Возвращает (gray_clean, steps) где steps — словарь промежуточных
    результатов для debug-режима."""
    steps = {}
    gray = to_gray(img)
    steps["gray"] = gray.copy()

    work = gray
    if cfg.subtract_background:
        work = subtract_background(work, cfg)
        steps["bg_subtracted"] = work.copy()
    if cfg.use_clahe:
        work = apply_clahe(work, cfg)
        steps["clahe"] = work.copy()
    work = denoise(work, cfg)
    steps["denoised"] = work.copy()

    return work, steps
