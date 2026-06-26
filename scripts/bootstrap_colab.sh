#!/usr/bin/env bash
# Bootstrap для Google Colab: распаковать zip -> поставить зависимости -> запустить обучение.
#
# ВАЖНО: Google Drive нужно примонтировать ДО запуска (из ячейки ноутбука Python):
#     from google.colab import drive; drive.mount('/content/drive')
#
# Пример ячейки Colab:
#     !bash /content/praktikum/scripts/bootstrap_colab.sh \
#         --zip "/content/praktikum_colab_*.zip" \
#         --work /content/praktikum \
#         --drive-runs /content/drive/MyDrive/praktikum/runs \
#         --name qr_yolo_v1 --mode auto --epochs 100 --model yolo11n.pt
#
# Режимы (--mode):
#   auto     — есть last.pt на Drive -> resume, иначе fresh   (по умолчанию)
#   fresh    — обучение с нуля
#   resume   — продолжить прерванное
#   finetune — дообучить best.pt (предполагает, что датасет дополнен)
set -euo pipefail

ZIP_GLOB=""
WORK="/content/praktikum"
DRIVE_RUNS="/content/drive/MyDrive/praktikum/runs"
NAME="qr_yolo_v1"
MODE="auto"
EPOCHS="100"
MODEL="yolo11n.pt"
EXTRA=()   # доп. флаги, прокидываются в train.py после '--'

while [[ $# -gt 0 ]]; do
  case "$1" in
    --zip)         ZIP_GLOB="$2"; shift 2;;
    --work)        WORK="$2"; shift 2;;
    --drive-runs)  DRIVE_RUNS="$2"; shift 2;;
    --name)        NAME="$2"; shift 2;;
    --mode)        MODE="$2"; shift 2;;
    --epochs)      EPOCHS="$2"; shift 2;;
    --model)       MODEL="$2"; shift 2;;
    --)            shift; EXTRA=("$@"); break;;
    *) echo "Неизвестный аргумент: $1" >&2; exit 1;;
  esac
done

# 1) распаковка (если задан --zip)
if [[ -n "$ZIP_GLOB" ]]; then
  ZIP_FILE="$(ls -t $ZIP_GLOB 2>/dev/null | head -1 || true)"
  [[ -z "$ZIP_FILE" ]] && { echo "zip не найден по маске: $ZIP_GLOB" >&2; exit 1; }
  echo ">> распаковываю $ZIP_FILE -> $WORK"
  mkdir -p "$WORK"
  unzip -o -q "$ZIP_FILE" -d "$WORK"
fi
cd "$WORK"

# 2) зависимости (для обучения достаточно ultralytics + pyyaml)
echo ">> ставлю зависимости (ultralytics, pyyaml)"
pip install -q ultralytics pyyaml

# 3) персистентность runs на Drive
mkdir -p "$DRIVE_RUNS"

# 4) авто-выбор режима
if [[ "$MODE" == "auto" ]]; then
  if [[ -f "$DRIVE_RUNS/$NAME/weights/last.pt" ]]; then
    MODE="resume"
  else
    MODE="fresh"
  fi
  echo ">> auto -> $MODE"
fi

# 5) запуск train.py с конкретными флагами
echo ">> старт обучения: mode=$MODE name=$NAME runs=$DRIVE_RUNS"
case "$MODE" in
  fresh)
    python src/train.py --mode fresh --data data.yaml --model "$MODEL" \
      --epochs "$EPOCHS" --project "$DRIVE_RUNS" --name "$NAME" "${EXTRA[@]}"
    ;;
  resume)
    python src/train.py --mode resume \
      --project "$DRIVE_RUNS" --name "$NAME" "${EXTRA[@]}"
    ;;
  finetune)
    python src/train.py --mode finetune --data data.yaml \
      --weights "$DRIVE_RUNS/$NAME/weights/best.pt" \
      --epochs "$EPOCHS" --lr0 0.001 \
      --project "$DRIVE_RUNS" --name "${NAME}_ft" "${EXTRA[@]}"
    ;;
  *) echo "Неизвестный MODE: $MODE" >&2; exit 1;;
esac
