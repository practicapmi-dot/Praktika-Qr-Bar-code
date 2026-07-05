"""barcode_normalizer — нормализация (выпрямление) 1D штрих-кодов.

Классический CV на OpenCV: сегментация зоны полос, оценка ориентации,
коррекция перспективы гомографией, апскейл. Полосы на выходе вертикальны.
"""
from .config import NormalizerConfig
from .normalizer import normalize_barcode, batch_normalize

__all__ = ["NormalizerConfig", "normalize_barcode", "batch_normalize"]
