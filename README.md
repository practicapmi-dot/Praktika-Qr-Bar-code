# Praktika — детекция и декодирование QR / штрихкодов (YOLO → crop → decode)

Система находит на изображении/видео коды (**QR, 1D-штрихкоды, DataMatrix, PDF417, Aztec**)
детектором **YOLO11**, вырезает каждый бокс и **декодирует** (`zxing-cpp`). Детектор обучается
на Google Colab (GPU); инференс — локально (CPU) или в Colab.

> Подробный контекст/архитектура — в `CLAUDE.md`, график работ по фазам — в `ROADMAP.md`
> (оба файла храним локально, в репозиторий не пушим).

## Что сделано

- ✅ **Датасет**: исходный VOC (952 картинки barcode/qr) сконвертирован в YOLO
  (`datasets/v1`: train 761 / val 95 / test 96). Классы в реале: `qr` + `barcode_1d`.
- ✅ **Обучение** на Colab: гибкий `train.py` (`fresh` / `resume` / `finetune`),
  персистентность на Google Drive. Обучена baseline-модель `best.pt` (yolo11n, 5 классов).
- ✅ **Инференс на видео + метрики**: `src/infer_video.py` (детекция + decode-first).
- ⏳ В планах (см. `ROADMAP.md`): нормализация кропов (ORB/deskew, Ph4),
  синтетика для `datamatrix/pdf417/aztec`, отдельные `detect/crop/normalize/decode/pipeline`.

## Какой файл за что отвечает

| Файл | Назначение |
|------|-----------|
| `data.yaml` | конфиг датасета YOLO: пути + 5 классов (`qr, barcode_1d, datamatrix, pdf417, aztec`) |
| `requirements.txt` | зависимости (ultralytics, opencv …; декодеры/синтетика — закомментированы) |
| `configs/train.yaml` | дефолты гиперпараметров для `train.py` |
| `src/train.py` | обучение YOLO: режимы `fresh` / `resume` / `finetune` |
| `src/infer_video.py` | инференс на видео + метрики (детекции, coverage, decode-rate); пишет аннотир. видео и `report.json` |
| `scripts/prepare_dataset.py` | VOC → YOLO: ремап классов, сплит train/val/test, `dataset_manifest.json` |
| `scripts/make_colab_zip.sh` | упаковка проекта (+датасет) в zip для Colab |
| `scripts/bootstrap_colab.sh` | в Colab: распаковка → install → авто-выбор режима → запуск обучения |
| `sem_17_03_2026.ipynb` | референс-ноутбук ORB (идея для будущей нормализации, Ph4) |

## Установка (локально)

```bash
python3 -m venv .venv
.venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install ultralytics zxing-cpp
# либо: .venv/bin/pip install -r requirements.txt   (+ zxing-cpp для декода)
```

## Как запускать

### 1. Подготовка датасета (VOC → YOLO)
```bash
python scripts/prepare_dataset.py --src datasets/raw/barcode_qr --out datasets/v1 \
    --train 0.8 --val 0.1 --test 0.1 --seed 0 --clean
```

### 2. Обучение
**Colab (рекомендуется — бесплатный GPU):**
```python
!git clone https://github.com/practicapmi-dot/Praktika-Qr-Bar-code.git
%cd Praktika-Qr-Bar-code
from google.colab import drive; drive.mount('/content/drive')
!bash scripts/bootstrap_colab.sh --work /content/Praktika-Qr-Bar-code \
    --drive-runs /content/drive/MyDrive/praktikum/runs --name qr_yolo_v1 --mode auto --epochs 100
```
Отвалился Colab → повтори ячейку: `--mode auto` сам сделает `resume` с `last.pt` на Drive.
Дообучение на дополненном датасете — тот же скрипт с `--mode finetune`.

**Локально:**
```bash
.venv/bin/python src/train.py --mode fresh --data data.yaml --model yolo11n.pt \
    --epochs 100 --imgsz 640 --batch 16 --name qr_yolo_v1
```

### 3. Инференс на видео + метрики
```bash
.venv/bin/python src/infer_video.py --weights best.pt --source OZONVIDEOS \
    --out runs/video --conf 0.25 --imgsz 640 --stride 5 --log-dets
```
Флаги: `--stride N` — обрабатывать каждый N-й кадр; `--decode` — считать decode-rate (zxing-cpp);
`--no-video` — без аннотированного видео; `--conf` — порог уверенности.

**Фильтр качества (sharp / blurry).** Каждый найденный кроп оценивается на читаемость:
резкость (дисперсия Лапласиана), контраст (std яркости) и минимальная сторона бокса.
Кроп «sharp», если `lap_var >= --blur-thr` **и** `contrast >= --min-contrast` **и**
`min(w,h) >= --min-box`. В аннотированном видео размытые боксы серые с меткой `BLUR`,
чёткие — в цвете класса. Флаги: `--blur-thr` (дефолт 1500 — откалиброван визуально по
OZONVIDEOS: ниже штрихи смазаны motion blur'ом), `--min-contrast` (25),
`--min-box` (24 px), `--only-sharp` — декодировать только sharp, `--log-dets` — писать
`detections.jsonl` с метриками каждого кропа (для калибровки порога под свои видео).
В `report.json` — блок `quality` (sharp/blurry counts) и, при `--decode`,
`decode.by_quality` (decode-rate отдельно по sharp и blurry).

### 4. Уникальные кропы через трекинг (`src/track_crops.py`)
```bash
.venv/bin/python src/track_crops.py --weights best.pt --source OZONVIDEOS \
    --out runs/track_crops --stride 5 --bin-thr 0
```
Один физический код = **один кроп за появление в кадре** (ByteTrack: ушёл из кадра →
вернулся → новый track id → новый кроп). Пока объект трекается, копится его самый резкий
кадр; когда трек ушёл (`--gone-after`, дефолт 15 обработанных кадров) — лучший кроп
бинаризуется и коммитится. Размытые треки отбрасываются (`--keep-blurry` — оставить).

Бинаризация `--bin-thr`: фиксированный порог (`pixel > thr → белый`, дефолт 210) или
**`0` = Otsu (авто-порог) — рекомендуется**: на OZONVIDEOS этикетки темнее 210 и
фиксированный порог заливает кроп чёрным, Otsu даёт чистые чёрные штрихи на белом.

Выход: `crops.npz` — массив ч/б кропов (uint8, 0/255), `crops_meta.json` — метаданные
по индексу, `crops/*.png` — просмотр глазами, `*_annot.mp4` — видео с track id.
Перебор в своём коде:
```python
import numpy as np, json
data = np.load("runs/track_crops/crops.npz")
crops = [data[k] for k in data.files]                      # список ч/б кропов
meta = json.load(open("runs/track_crops/crops_meta.json")) # meta[i] ↔ crops[i]
```

### 5. Полный пайплайн: трекинг → нормализация (`src/pipeline_video.py`)
```bash
.venv/bin/python src/pipeline_video.py --weights best.pt --source OZONVIDEOS \
    --out runs/pipeline --stride 5 --bin-thr 0
```
Этап 1 — `track_crops` (всё из раздела 4). Этап 2 — каждый **сырой** кроп из массива
проходит через `src/barcode_normalizer/` (пакет из `normalizer_05.07.zip`: ориентация
штрихов → поворот в вертикаль → сегментация зоны → гомография → апскейл) и сохраняется
в отдельную папку `normalized/` (+ `normalized.npz`, индексы совпадают с `crops/`).
Нормализатор рассчитан на 1D-коды: по умолчанию выпрямляются только `barcode_1d`
(`--norm-classes`), остальные классы копируются как есть. Доп. флаги:
`--target-height` (256), `--norm-gray` (одноканальный выход), `--no-perspective`.

## Где видео и где результаты

- **Входные видео:** `OZONVIDEOS/` — 3 ролика Ozon (4K @ 20fps, ~62 c). **Проприетарные →
  в git НЕ коммитятся** (`.gitignore`), лежат только локально.
- **Веса:** `best.pt` в корне (скачаны из Colab; `*.pt` в `.gitignore`).
- **Результаты инференса:** `runs/video/` — аннотированные `*_annot.mp4` + `report.json` (все метрики).
  `runs/` в `.gitignore` (генерируется, хранится локально).

## Метрики — baseline `best.pt` на OZONVIDEOS

Прогон: `conf 0.25`, `imgsz 640`, `stride 5` (≈4 fps), decode-first (zxing-cpp по сырому кропу).

| Видео | Кадров | Детекций (qr / barcode_1d) | Coverage | Decoded |
|-------|--------|----------------------------|----------|---------|
| палет 1 | 251 | 607 (599 / 8) | 94.4% | 0 / 607 |
| палет 2 | 251 | 3976 (3976 / 0) | 99.2% | 0 / 3976 |
| стол | 251 | 77 (75 / 2) | 19.9% | 0 / 77 |
| **ИТОГО** | **753** | **4660 (4650 / 10)** | **71.2%** | **0 / 4660 (0%)** |

Средняя уверенность: `qr` 0.527, `barcode_1d` 0.292.

> На сыром видео нет ground-truth разметки, поэтому mAP/precision/recall не считаются —
> приведены операционные метрики.

**Интерпретация (честно):** детектор **срабатывает** и садится на транспортные этикетки коробок,
но **end-to-end decode = 0%**. Причины: (1) домен-сдвиг — обучение шло на ручных фото товаров,
а тут склад сверху; (2) модель почти всё зовёт `qr` и переусердствует (видео-2: ~16 боксов/кадр),
боксы рыхлые/наложенные; (3) коды мелкие на расстоянии даже в 4K; (4) **стадия нормализации
(Ph4) ещё не реализована** — декод идёт по сырому кропу. Сам декодер рабочий
(zxing декодит 22/50 датасетных картинок).

**Что поднимет decode-rate:** дообучение на складских данных (`--mode finetune`); нормализация
кропа (upscale + grayscale + threshold + deskew + повороты 0/90/180/270); калибровка классов
(на складе в основном 1D/DataMatrix, а не `qr`).

## Датасет

- `datasets/raw/barcode_qr/` — исходный VOC (952 `jpg` + `xml`).
- `datasets/v1/` — YOLO-формат (`images/` и `labels/` × `train/val/test`) + `dataset_manifest.json`.
- Классы (5): `qr, barcode_1d, datamatrix, pdf417, aztec`. Сейчас в данных только `qr` и
  `barcode_1d`; остальные три — под синтетику (следующий шаг).
