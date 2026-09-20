#!/usr/bin/env bash
# Полный цикл воспроизведения результатов НИР.
# Использование: bash run_all.sh
set -e
cd "$(dirname "$0")"

echo "=== 1/4. Расчёты (стадии A-H) ==="
cd src
python3 -u run_experiments.py H A B C D E F G
python3 -u - <<'PY'
from run_experiments import base, run
run(base(name="R1_res128", x_name="lr0", y_name="lr1", resolution=128),
    "исследование зависимости от разрешения, 128x128")
PY

echo "=== 2/4. Рисунки ==="
python3 make_figures.py

echo "=== 3/4. Таблицы и числа для отчёта ==="
python3 make_tables.py
cd ..

echo "=== 4/4. Сборка отчёта ==="
cd report
xelatex -interaction=nonstopmode main.tex >/dev/null
xelatex -interaction=nonstopmode main.tex >/dev/null
echo "Готово: report/main.pdf"
