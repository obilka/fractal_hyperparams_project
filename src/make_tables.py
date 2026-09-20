"""
make_tables.py — генерация LaTeX-таблиц и макросов с числами
непосредственно из посчитанных данных (outputs/data/*.json).

Запуск: python3 make_tables.py
Результат: report/tables/*.tex и report/tables/numbers.tex
"""
from __future__ import annotations

import json
import os

import numpy as np

from fractal_hp import DATA_DIR

TAB_DIR = os.path.join(os.path.dirname(__file__), "..", "report", "tables")
os.makedirs(TAB_DIR, exist_ok=True)


def meta(name):
    p = os.path.join(DATA_DIR, f"{name}.json")
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def fD(m):
    if m is None or not np.isfinite(m["D"]):
        return "---"
    return f"${m['D']:.3f} \\pm {m['sigma']:.3f}$"


def fr(m, key):
    return "---" if m is None else f"{m[key]:.2f}"


def write(fname, text):
    with open(os.path.join(TAB_DIR, fname), "w", encoding="utf-8") as f:
        f.write(text)
    print("table:", fname)


HEAD4 = (r"\begin{tabular}{lccccc}" "\n" r"\toprule" "\n"
         r"Конфигурация & $D_{\mathrm{box}}$ & сход. & осц. & расх. & NaN\\" "\n"
         r"\midrule" "\n")
FOOT = r"\bottomrule" "\n" r"\end{tabular}" "\n"


def row(label, name):
    m = meta(name)
    return (f"{label} & {fD(m)} & {fr(m,'conv_frac')} & {fr(m,'osc_frac')} & "
            f"{fr(m,'div_frac')} & {fr(m,'nan_frac')}\\\\\n")


def tab_generic(fname, rows):
    write(fname, HEAD4 + "".join(row(l, n) for l, n in rows) + FOOT)


def tab_zoom():
    ds, rows = [], ""
    for k in range(5):
        m = meta(f"F_zoom{k}")
        if m is None:
            continue
        c = m["config"]
        ds.append(m["D"])
        rows += (f"{k} & $[{c['x_range'][0]:.3g},\\,{c['x_range'][1]:.3g}]$ & "
                 f"$[{c['y_range'][0]:.3g},\\,{c['y_range'][1]:.3g}]$ & "
                 f"${m['D']:.3f} \\pm {m['sigma']:.3f}$\\\\\n")
    if not ds:
        return None, None
    med, sd = float(np.median(ds)), float(np.std(ds))
    t = (r"\begin{tabular}{cccc}" "\n" r"\toprule" "\n"
         r"Уровень & Диапазон $\eta$ & Диапазон $\beta$ & $D_{\mathrm{box}}$\\" "\n"
         r"\midrule" "\n" + rows + r"\midrule" "\n"
         rf"\multicolumn{{3}}{{c}}{{\textbf{{Медиана}}}} & $\mathbf{{{med:.3f}}}$\\" "\n"
         rf"\multicolumn{{3}}{{c}}{{Стандартное отклонение}} & ${sd:.3f}$\\" "\n" + FOOT)
    write("tab_zoom.tex", t)
    return med, sd


def numbers(med, sd):
    """Макросы с числами для использования в тексте отчёта."""
    mapping = {
        "DBaseTanh": "A1_base_tanh", "DBaseRelu": "A2_base_relu",
        "DBaseLinear": "A3_base_linear", "DEtaBeta": "B1_eta_beta",
        "DEtaWd": "B2_eta_wd", "DMbOne": "B4_mb_B1", "DMbFour": "B4_mb_B4",
        "DMbFull": "B4_mb_B32", "DSgd": "C1_opt_sgd",
        "DMomentum": "C2_opt_momentum", "DAdam": "C3_opt_adam",
        "DRmsprop": "C4_opt_rmsprop", "DAdamBOne": "C5_adam_eta_b1",
        "DAdamBTwo": "C6_adam_eta_b2", "DSynth": "D1_synth16",
        "DDigits": "D2_digits16", "DWdZero": "E0_eta_beta_wd0",
        "DWdOne": "E1_eta_beta_wd1", "DWdTwo": "E2_eta_beta_wd2",
        "DWdThree": "E3_eta_beta_wd3", "DFloatThirtyTwo": "G1_float32", "DTanhMb": "A6_tanh_mb",
        "DReluMbLow": "A7_relu_mb_low", "DReluMb": "A5_relu_mb",
        "DLinearWide": "A4_linear_wide", "DResSmall": "R1_res128",
    }
    out = []
    for macro, name in mapping.items():
        m = meta(name)
        v = "---" if m is None or not np.isfinite(m["D"]) else f"{m['D']:.3f}"
        s = "" if m is None or not np.isfinite(m["D"]) else f"{m['sigma']:.3f}"
        out.append(rf"\newcommand{{\{macro}}}{{{v}}}")
        out.append(rf"\newcommand{{\{macro}S}}{{{s}}}")
        if m is not None:
            out.append(rf"\newcommand{{\{macro}Conv}}{{{m['conv_frac']:.2f}}}")
    if med is not None:
        out.append(rf"\newcommand{{\DZoomMedian}}{{{med:.3f}}}")
        out.append(rf"\newcommand{{\DZoomStd}}{{{sd:.3f}}}")
    # суммарное машинное время
    tot = 0.0
    for f in os.listdir(DATA_DIR):
        if f.endswith(".json") and f != "F_zoom_center.json":
            m = json.load(open(os.path.join(DATA_DIR, f), encoding="utf-8"))
            tot += m.get("seconds", 0.0)
    out.append(rf"\newcommand{{\TotalHours}}{{{tot/3600:.1f}}}")
    write("numbers.tex", "\n".join(out) + "\n")


def tab_activations_ext():
    """Расширенная таблица активаций с указанием окна сканирования."""
    rows = [("tanh", "A1_base_tanh", r"$[10^{-2},10]$, $|\mathcal{D}|=1$, $512^2$"),
            ("ReLU", "A2_base_relu", r"$[10^{-2},10]$, $|\mathcal{D}|=1$"),
            ("линейная", "A3_base_linear", r"$[10^{-2},10]$, $|\mathcal{D}|=1$"),
            ("линейная", "A4_linear_wide", r"$[10^{-4},10^{-1}]$, $|\mathcal{D}|=1$"),
            ("tanh", "A6_tanh_mb", r"$[10^{-1},10^{3}]$, $|\mathcal{D}|=32$"),
            ("ReLU", "A7_relu_mb_low", r"$[10^{-3},10]$, $|\mathcal{D}|=32$"),
            ("ReLU", "A5_relu_mb", r"$[10^{-1},10^{3}]$, $|\mathcal{D}|=32$")]
    t = (r"\begin{tabular}{llccccc}" "\n" r"\toprule" "\n"
         r"Активация & Окно по $\eta$ & $D_{\mathrm{box}}$ & сход. & осц. & расх. & NaN\\" "\n"
         r"\midrule" "\n")
    for lab, name, win in rows:
        m = meta(name)
        t += (f"{lab} & {win} & {fD(m)} & {fr(m,'conv_frac')} & {fr(m,'osc_frac')} & "
              f"{fr(m,'div_frac')} & {fr(m,'nan_frac')}\\\\\n")
    write("tab_activations_ext.tex", t + FOOT)


def tab_resolution():
    rows = ""
    for lab, name in [("$128\\times128$", "R1_res128"),
                      ("$256\\times256$", "C1_opt_sgd"),
                      ("$512\\times512$", "A1_base_tanh")]:
        m = meta(name)
        if m is None:
            continue
        rows += f"{lab} & {fD(m)} & {fr(m,'conv_frac')}\\\\\n"
    if not rows:
        return
    write("tab_resolution.tex",
          r"\begin{tabular}{lcc}" "\n" r"\toprule" "\n"
          r"Разрешение сетки & $D_{\mathrm{box}}$ & доля сходимости\\" "\n"
          r"\midrule" "\n" + rows + FOOT)


def main():
    tab_activations_ext()
    tab_resolution()
    tab_generic("tab_sections.tex", [
        (r"$(\eta_0,\eta_1)$, SGD", "C1_opt_sgd"),
        (r"$(\eta,\beta)$, momentum", "B1_eta_beta"),
        (r"$(\eta,\lambda_{wd})$, SGD", "B2_eta_wd")])
    tab_generic("tab_minibatch.tex", [
        ("$B=1$", "B4_mb_B1"), ("$B=4$", "B4_mb_B4"),
        ("$B=32$ (полный батч)", "B4_mb_B32")])
    tab_generic("tab_optimizers.tex", [
        ("SGD", "C1_opt_sgd"), (r"SGD+momentum ($\beta=0.9$)", "C2_opt_momentum"),
        ("Adam", "C3_opt_adam"), ("RMSprop", "C4_opt_rmsprop"),
        (r"Adam, сечение $(\eta,\beta_1)$", "C5_adam_eta_b1"),
        (r"Adam, сечение $(\eta,\beta_2)$", "C6_adam_eta_b2")])
    tab_generic("tab_data.tex", [
        ("случайные данные, $|D|=16$", "D1_synth16"),
        ("digits (PCA-16), $|D|=16$", "D2_digits16")])
    tab_generic("tab_wd_slices.tex", [
        (r"$\lambda_{wd}=0$", "E0_eta_beta_wd0"),
        (r"$\lambda_{wd}=10^{-4}$", "E1_eta_beta_wd1"),
        (r"$\lambda_{wd}=10^{-3}$", "E2_eta_beta_wd2"),
        (r"$\lambda_{wd}=10^{-2}$", "E3_eta_beta_wd3")])
    tab_generic("tab_precision.tex", [
        ("float64", "E0_eta_beta_wd0"), ("float32", "G1_float32")])
    med, sd = tab_zoom()
    numbers(med, sd)


if __name__ == "__main__":
    main()
