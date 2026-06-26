#!/usr/bin/env bash
# Упаковка проекта в zip для заливки в Google Colab.
# По умолчанию ВКЛЮЧАЕТ датасет (чтобы обучать прямо в Colab). Отключить: --no-data.
#
# Примеры:
#   scripts/make_colab_zip.sh                  # всё + датасет -> dist/
#   scripts/make_colab_zip.sh --no-data        # без датасета (только код)
#   scripts/make_colab_zip.sh -n qr -o /tmp    # своё имя/папка
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

OUT_DIR="$ROOT_DIR/dist"
NAME="praktikum_colab"
WITH_DATA=1
DATA_DIR="datasets"

usage() {
  cat <<EOF
Использование: scripts/make_colab_zip.sh [опции]
  -o, --out DIR     куда класть zip (default: dist/)
  -n, --name NAME   базовое имя архива (default: praktikum_colab)
      --no-data     не включать датасет ($DATA_DIR/)
      --data-dir D  папка датасета (default: datasets)
  -h, --help        помощь
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -o|--out)      OUT_DIR="$2"; shift 2;;
    -n|--name)     NAME="$2"; shift 2;;
    --no-data)     WITH_DATA=0; shift;;
    --data-dir)    DATA_DIR="$2"; shift 2;;
    -h|--help)     usage; exit 0;;
    *) echo "Неизвестный аргумент: $1" >&2; usage; exit 1;;
  esac
done

command -v zip >/dev/null 2>&1 || {
  echo "Ошибка: нет утилиты 'zip'. Установи: sudo apt-get install -y zip" >&2
  exit 1
}

cd "$ROOT_DIR"
mkdir -p "$OUT_DIR"
TS="$(date +%Y%m%d_%H%M%S)"
ZIP_PATH="$OUT_DIR/${NAME}_${TS}.zip"

# Включаем только существующие пути, нужные для обучения в Colab:
CANDIDATES=(src scripts configs data.yaml requirements.txt notebooks)
INCLUDE=()
for c in "${CANDIDATES[@]}"; do
  [[ -e "$c" ]] && INCLUDE+=("$c")
done
if [[ "$WITH_DATA" -eq 1 ]]; then
  if [[ -d "$DATA_DIR" ]]; then
    INCLUDE+=("$DATA_DIR")
  else
    echo "Предупреждение: датасет '$DATA_DIR/' не найден — пропускаю (соберётся в Ph1)." >&2
  fi
fi

if [[ ${#INCLUDE[@]} -eq 0 ]]; then
  echo "Ошибка: нечего архивировать (нет ни src/, ни data.yaml...)." >&2
  exit 1
fi

# Исключения (кэши/мусор/веса):
EXCLUDES=( "*/__pycache__/*" "*.pyc" "*.pyo" "*/.git/*"
           "*/.ipynb_checkpoints/*" "*/runs/*" "*.DS_Store" )
EX_ARGS=()
for e in "${EXCLUDES[@]}"; do EX_ARGS+=(-x "$e"); done

echo "Архивирую: ${INCLUDE[*]}"
zip -r -q "$ZIP_PATH" "${INCLUDE[@]}" "${EX_ARGS[@]}"

SIZE="$(du -h "$ZIP_PATH" | cut -f1)"
COUNT="$(unzip -l "$ZIP_PATH" | tail -1 | awk '{print $2}')"
echo "Готово: $ZIP_PATH  (размер: $SIZE, файлов: $COUNT)"
echo "Залей этот zip в Colab и запусти scripts/bootstrap_colab.sh (см. его шапку)."
