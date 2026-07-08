# Praktika — детекция, нормализация и декодирование QR / штрихкодов

Система находит коды на видео (**QR, 1D-штрихкоды, DataMatrix, PDF417, Aztec**) детектором
**YOLO11**, трекает каждый физический код (**ByteTrack**), отбирает его самый резкий кадр,
вырезает кроп, **нормализует** (выпрямление штрихов + перспектива) и **декодирует** (pyzbar /
zxing-cpp). Детектор обучается на Google Colab (GPU); инференс — локально (CPU).

```
видео → YOLO11 детекция → ByteTrack трекинг → фильтр резкости (Лаплас)
      → лучший кроп на трек → barcode_normalizer (угол → поворот → сегментация
      → гомография → апскейл) → [бинаризация] → pyzbar decode → JSON
```

## Как устроен репозиторий

| Файл / папка | Назначение |
|------|-----------|
| `src/pipeline_video.py` | **главный скрипт**: видео → уникальные кропы → нормализация → декодирование |
| `src/track_crops.py` | этап 1 отдельно: YOLO+ByteTrack, один лучший кроп на появление кода в кадре |
| `src/infer_video.py` | инференс с метриками (детекции, coverage, decode-rate, sharp/blurry) |
| `src/barcode_normalizer/` | пакет нормализации 1D-кодов (конфиг — `config.py`, все параметры с комментариями) |
| `src/train.py` | обучение YOLO: режимы `fresh` / `resume` / `finetune` |
| `scripts/prepare_dataset.py` | VOC → YOLO: ремап классов, сплит train/val/test, манифест |
| `scripts/bootstrap_colab.sh` | Colab: установка → авто-выбор режима → запуск обучения |
| `scripts/make_colab_zip.sh` | упаковка проекта в zip для Colab (альтернатива git clone) |
| `scripts/synth_bench.py` | синтетический decode-бенчмарк нормализатора (главная метрика качества) |
| `scripts/eval_normalizer.py` | метрики нормализатора на реальных кропах пайплайна |
| `scripts/param_search.py` | random search параметров `NormalizerConfig` по decode-rate |
| `data.yaml` | конфиг датасета YOLO: пути + 5 классов (`qr, barcode_1d, datamatrix, pdf417, aztec`) |
| `configs/train.yaml` | дефолты гиперпараметров обучения |
| `datasets/`, `runs/`, `*.pt` | данные / результаты / веса — в `.gitignore`, живут локально |

## Установка

```bash
git clone https://github.com/practicapmi-dot/Praktika-Qr-Bar-code.git
cd Praktika-Qr-Bar-code
python3 -m venv .venv
.venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -r requirements.txt
```

Для pyzbar нужна системная библиотека: `sudo apt install libzbar0` (на Kali/Ubuntu часто уже есть).
Веса `best.pt` в git не хранятся — скачай из Colab/Drive и положи в корень репозитория.

## Быстрый старт — полный пайплайн

```bash
.venv/bin/python src/pipeline_video.py --weights best.pt --source OZONVIDEOS \
    --out runs/pipeline --stride 5 --bin-thr 0
```

Результаты в `runs/pipeline/`:

| Артефакт | Что это |
|----------|---------|
| `crops/*.png`, `crops.npz` | уникальные бинаризованные ч/б кропы (этап 1) |
| `crops_meta.json` | по индексу: видео, кадр, track_id, класс, резкость, bbox |
| `normalized/*.png`, `normalized.npz` | выпрямленные кропы (этап 2); индексы совпадают с `crops/` |
| `decoded.json` | этап 3: результат pyzbar по каждому кропу (`text: null` = не прочитан) |
| `*_annot.mp4` | видео с track id (`CAP` = кроп зафиксирован) |

Перебор массивов в своём коде:
```python
import numpy as np, json
data = np.load("runs/pipeline/normalized.npz")
crops = [data[k] for k in data.files]
meta = json.load(open("runs/pipeline/crops_meta.json"))   # meta[i] ↔ crops[i]
```

## Все флаги

### `pipeline_video.py` (включает все флаги `track_crops.py`)

| Флаг | Дефолт | Описание |
|------|--------|----------|
| `--weights` | `best.pt` | веса YOLO |
| `--source` | — | видеофайл или папка с видео |
| `--out` | `runs/pipeline` | папка результатов |
| `--conf` | 0.25 | порог уверенности детектора |
| `--imgsz` | 640 | размер инференса YOLO |
| `--stride` | 5 | обрабатывать каждый N-й кадр (меньше = надёжнее трекинг, медленнее) |
| `--pad` | 0.10 | запас вокруг BB при кропе (+10% с каждой стороны) |
| `--max-frames` | 0 | лимит обработанных кадров (0 = все) |
| `--bin-thr` | 210 | бинаризация кропов этапа 1: фикс. порог; **`0` = Otsu (рекомендуется)** |
| `--blur-thr` | 1500 | порог резкости (дисперсия Лапласиана); ниже — blurry |
| `--min-contrast` | 25 | мин. контраст кропа (std яркости) |
| `--min-box` | 24 | мин. сторона бокса, px |
| `--gone-after` | 15 | через сколько кадров без трека коммитить его кроп |
| `--keep-blurry` | выкл | не отбрасывать треки, не прошедшие фильтр резкости |
| `--no-video` | выкл | не писать аннотированное видео |
| `--annot-width` | 1280 | ширина аннотированного видео |
| `--device` | `cpu` | `cpu` / `0` (GPU) |
| `--target-height` | 256 | высота выпрямленного кода после нормализации |
| `--norm-classes` | `barcode_1d` | какие классы нормализовать (через запятую) |
| `--norm-gray` | выкл | нормализованный выход одноканальный |
| `--norm-binary` | **выкл** | режим с бинаризацией финала (ч/б 0/255 по Otsu) |
| `--no-perspective` | выкл | отключить коррекцию перспективы |
| `--decode-scales` | `1,2,3` | масштабы попыток декодирования (pyzbar чувствителен к размеру штриха) |

**Два режима финала:** по умолчанию `normalized/` — **без бинаризации** (декодеры бинаризуют
сами, адаптивно — decode-rate выше; жёсткий глобальный порог терял читаемые коды).
`--norm-binary` — финал строго ч/б 0/255 (для хранения/попиксельной обработки); при
декодировании автоматически используется fallback на небинаризованную версию.

### `infer_video.py` (метрики детекции)

Те же базовые флаги + `--decode` (считать decode-rate zxing-ом), `--only-sharp` (декодировать
только резкие), `--log-dets` (писать `detections.jsonl` с `lap_var`/`contrast` каждого кропа —
для калибровки `--blur-thr` под свои видео).

### Бенчмарки нормализатора

```bash
.venv/bin/python scripts/synth_bench.py --cases 120 --variants all   # синтетика: decode-rate
.venv/bin/python scripts/eval_normalizer.py                          # реальные кропы
.venv/bin/python scripts/param_search.py --n 64 --cases 100 --workers 8   # поиск параметров
```

`param_search` пишет `runs/param_search/results.jsonl` (все конфиги) и `best_config.json`
(лучший, валидирован на втором сиде). Найденные значения уже вшиты в
`src/barcode_normalizer/config.py`.

## Обучение модели

### Подготовка датасета

```bash
.venv/bin/python scripts/prepare_dataset.py --src datasets/raw/barcode_qr --out datasets/v1 \
    --train 0.8 --val 0.1 --test 0.1 --seed 0 --clean
```
Флаги: `--version-tag` (тег в манифест), `--no-dedup` (не выкидывать дубли), `--symlink`
(симлинки вместо копий), `--clean` (очистить `--out`).

### Три режима `train.py`

| Режим | Когда | Что делает |
|-------|-------|-----------|
| `fresh` | первое обучение | старт с COCO-весов (`--model yolo11n.pt`) |
| `resume` | Colab отвалился / прервал | продолжает run с `last.pt` **вместе с состоянием оптимизатора** — с той же эпохи |
| `finetune` | пополнил датасет / новый домен | новый run от `best.pt` с пониженным LR |

```bash
# первое обучение
.venv/bin/python src/train.py --mode fresh --data data.yaml --model yolo11n.pt \
    --epochs 100 --imgsz 640 --batch 16 --name qr_yolo_v1

# продолжить прерванное (указать last.pt прерванного run'а)
.venv/bin/python src/train.py --mode resume --resume-path runs/qr_yolo_v1/weights/last.pt

# дообучение на новых данных (напр. складские кадры) с меньшим LR
.venv/bin/python src/train.py --mode finetune --weights best.pt --data datasets/v2/data.yaml \
    --epochs 50 --lr0 0.001 --name qr_yolo_v2_ft
```

Остальные флаги: `--config` (свой yaml с дефолтами), `--project` (куда писать runs),
`--patience` (early stopping), `--seed`, `--workers`, `--device` (`0`/`cpu`/`0,1`),
`--batch -1` (автоподбор).

### Обучение в Colab (рекомендуется — бесплатный GPU)

```python
!git clone https://github.com/practicapmi-dot/Praktika-Qr-Bar-code.git
%cd Praktika-Qr-Bar-code
from google.colab import drive; drive.mount('/content/drive')
!bash scripts/bootstrap_colab.sh --work /content/Praktika-Qr-Bar-code \
    --drive-runs /content/drive/MyDrive/praktikum/runs --name qr_yolo_v1 --mode auto --epochs 100
```

Ключевое — персистентность через Drive: чекпойнты пишутся туда каждую эпоху.
**Colab отвалился → просто повтори ячейку**: `--mode auto` сам найдёт `last.pt` на Drive и
сделает `resume`; если run завершён и есть новые данные — запускай с `--mode finetune`.
Флаги bootstrap: `--zip` (путь к архиву проекта вместо clone), `--work`, `--drive-runs`,
`--name`, `--mode {auto,fresh,resume,finetune}`, `--epochs`, `--model`.

Как пополнять датасет для finetune: добавить новые изображения+разметку в `datasets/raw/...`,
прогнать `prepare_dataset.py --out datasets/v2 --version-tag v2`, обучить `--mode finetune
--data datasets/v2/data.yaml`. Манифест (`dataset_manifest.json`) фиксирует состав каждой версии.

## Результаты на OZONVIDEOS (3 ролика 4K @ 20fps, склад)

**Пайплайн (полный):** 753 кадра → 162 трека → **124 уникальных кропа** (38 отброшено как
размытые) → нормализовано 124/124 → **декодировано pyzbar: 1/124** (`246582`).

**Нормализатор** (после улучшений + подбора параметров):

| Метрика | Оригинал | Улучшенный |
|---------|----------|-----------|
| Синтетика, decode-rate (120 кейсов) | 45.8% | **69.2%** |
| Реальные кропы, вертикальность штрихов | 0.829 | **0.882** |

Интерполяция проверена бенчем: Lanczos лучший (linear −20пп, cubic −3пп decode).

**Почему decode на реальном видео ~0:** коды в кадре ~50 px по меньшей стороне — один штрих
меньше пикселя, это ниже физического предела любого декодера (синтетика доказывает, что сама
цепочка декодирует ~70%, когда пикселей хватает). Решение — камера ближе/зум к зоне
сканирования; софтом это не лечится.

## Датасет

- `datasets/raw/barcode_qr/` — исходный VOC (952 `jpg` + `xml`).
- `datasets/v1/` — YOLO-формат + `dataset_manifest.json`; `datasets/v2/` — v1 + OZON-кадры + аугментации.
- Классы (5): `qr, barcode_1d, datamatrix, pdf417, aztec`. В данных пока `qr` и `barcode_1d`;
  остальные — под синтетику (`ROADMAP.md`).
