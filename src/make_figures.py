"""
make_figures.py — построение всех рисунков отчёта по сохранённым данным.
Запуск: python3 make_figures.py
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm, LinearSegmentedColormap

from fractal_hp import (DATA_DIR, FIG_DIR, load_result, extract_boundary,
                        CONVERGED, OSCILLATING, DIVERGED, NONFINITE)

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 8,
    "axes.titlesize": 9, "axes.labelsize": 8, "savefig.bbox": "tight",
})

REG_COLORS = ["#1b4f9c", "#f0a830", "#c0392b", "#101010"]
REG_LABELS = ["сходимость", "осцилляции / период.", "расходимость", "NaN / Inf"]
REG_CMAP = ListedColormap(REG_COLORS)
REG_NORM = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], REG_CMAP.N)

AXIS_LABEL = {
    "lr0": r"$\eta_0$ (входной слой)",
    "lr1": r"$\eta_1$ (выходной слой)",
    "lr": r"$\eta$ (шаг обучения)",
    "beta": r"$\beta$ (momentum)",
    "beta1": r"$\beta_1$ (Adam)",
    "beta2": r"$\beta_2$ (Adam)",
    "wd": r"$\lambda_{wd}$ (weight decay)",
    "batch": r"$B$ (размер батча)",
}


def meta(name):
    with open(os.path.join(DATA_DIR, f"{name}.json"), encoding="utf-8") as f:
        return json.load(f)


def has(name):
    return os.path.exists(os.path.join(DATA_DIR, f"{name}.npz"))


def _axes_setup(ax, m):
    c = m["config"]
    if c["x_log"]:
        ax.set_xscale("log")
    if c["y_log"]:
        ax.set_yscale("log")
    ax.set_xlabel(AXIS_LABEL.get(c["x_name"], c["x_name"]))
    ax.set_ylabel(AXIS_LABEL.get(c["y_name"], c["y_name"]))


def plot_regime(ax, name, title=None, show_axes=True):
    r = load_result(name)
    m = meta(name)
    ax.pcolormesh(r["xs"], r["ys"], r["regime"], cmap=REG_CMAP, norm=REG_NORM,
                  shading="auto", rasterized=True)
    if show_axes:
        _axes_setup(ax, m)
    if title is None:
        title = f"{m['note']}\n$D_{{box}}={m['D']:.3f}\\pm{m['sigma']:.3f}$"
    ax.set_title(title)
    return m


def plot_continuous(ax, name, title=None):
    """Сине-красная карта в стиле предыдущей НИР."""
    r = load_result(name)
    m = meta(name)
    reg, sl, si = r["regime"], r["sum_l"], r["sum_inv"]
    conv = reg == CONVERGED
    img = np.zeros(reg.shape + (3,))
    if conv.any():
        v = np.log10(np.maximum(sl[conv], 1e-12))
        v = (v - v.min()) / max(np.ptp(v), 1e-12)
        img[conv] = np.stack([0.15 + 0.7 * v, 0.35 + 0.55 * v,
                              0.75 + 0.25 * v], axis=-1)
    nc = ~conv
    if nc.any():
        v = np.log10(np.maximum(si[nc], 1e-12))
        v = (v - v.min()) / max(np.ptp(v), 1e-12)
        img[nc] = np.stack([0.75 + 0.25 * v, 0.2 + 0.6 * v,
                            0.15 + 0.6 * v], axis=-1)
    xs, ys = r["xs"], r["ys"]
    ax.imshow(img, origin="lower", aspect="auto",
              extent=[0, 1, 0, 1], interpolation="nearest")
    c = m["config"]
    nt = 5
    xi = np.linspace(0, 1, nt)
    ax.set_xticks(xi)
    ax.set_xticklabels([f"{v:.1e}" if c["x_log"] else f"{v:.2f}"
                        for v in np.interp(xi, [0, 1], [0, len(xs) - 1]).astype(int) * 0
                        + xs[np.linspace(0, len(xs) - 1, nt).astype(int)]],
                       rotation=30)
    ax.set_yticks(xi)
    ax.set_yticklabels([f"{v:.1e}" if c["y_log"] else f"{v:.2f}"
                        for v in ys[np.linspace(0, len(ys) - 1, nt).astype(int)]])
    ax.set_xlabel(AXIS_LABEL.get(c["x_name"], c["x_name"]))
    ax.set_ylabel(AXIS_LABEL.get(c["y_name"], c["y_name"]))
    ax.set_title(title or "суммарный loss: синий — сходимость, красный — расходимость")


def legend_fig(fig, ncol=4):
    handles = [plt.Line2D([], [], marker="s", ls="", ms=7, color=c,
                          label=l) for c, l in zip(REG_COLORS, REG_LABELS)]
    fig.tight_layout(rect=[0, 0.10, 1, 1])
    fig.legend(handles=handles, loc="lower center", ncol=ncol, frameon=False,
               bbox_to_anchor=(0.5, 0.005), fontsize=7.5)


def save(fig, fname):
    p = os.path.join(FIG_DIR, fname)
    fig.savefig(p)
    plt.close(fig)
    print("figure:", fname)


# ---------------------------------------------------------------- рисунки

def fig_base_tanh():
    if not has("A1_base_tanh"):
        return
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.4))
    m = plot_regime(axs[0], "A1_base_tanh",
                    f"карта режимов, tanh\n$D_{{box}}={meta('A1_base_tanh')['D']:.3f}"
                    f"\\pm{meta('A1_base_tanh')['sigma']:.3f}$")
    plot_continuous(axs[1], "A1_base_tanh")
    legend_fig(fig)
    save(fig, "f01_base_tanh.png")


def fig_activations():
    names = [n for n in ("A1_base_tanh", "A2_base_relu", "A3_base_linear") if has(n)]
    if not names:
        return
    fig, axs = plt.subplots(1, len(names), figsize=(3.0 * len(names), 3.2))
    axs = np.atleast_1d(axs)
    tt = {"A1_base_tanh": "tanh", "A2_base_relu": "ReLU", "A3_base_linear": "линейная"}
    for ax, n in zip(axs, names):
        m = meta(n)
        plot_regime(ax, n, f"{tt[n]}: $D={m['D']:.3f}\\pm{m['sigma']:.3f}$")
    legend_fig(fig)
    save(fig, "f02_activations.png")


def fig_boxcount():
    ns = [n for n in ("A1_base_tanh", "B1_eta_beta") if has(n)]
    if not ns:
        return
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    for n, mk in zip(ns, ["o", "s"]):
        m = meta(n)
        s = np.array(m["box_sizes"], float)
        c = np.array(m["box_counts"], float)
        ax.plot(np.log(s), np.log(c), mk, ms=5, label=f"{n}: D={m['D']:.3f}")
        k = np.polyfit(np.log(s), np.log(c), 1)
        xx = np.linspace(np.log(s).min(), np.log(s).max(), 10)
        ax.plot(xx, np.polyval(k, xx), "--", lw=1)
    ax.set_xlabel(r"$\log \varepsilon$")
    ax.set_ylabel(r"$\log N(\varepsilon)$")
    ax.set_title("Box-counting: линейность в log-log")
    ax.legend(fontsize=7)
    ax.grid(alpha=.3)
    save(fig, "f03_boxcounting.png")


def fig_eta_beta():
    if not has("B1_eta_beta"):
        return
    m = meta("B1_eta_beta")
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.4))
    plot_regime(axs[0], "B1_eta_beta",
                f"сечение $(\\eta,\\beta)$, SGD+momentum\n"
                f"$D_{{box}}={m['D']:.3f}\\pm{m['sigma']:.3f}$")
    r = load_result("B1_eta_beta")
    bnd = extract_boundary(r["regime"])
    axs[1].pcolormesh(r["xs"], r["ys"], bnd, cmap="binary", shading="auto",
                      rasterized=True)
    axs[1].set_xscale("log")
    axs[1].set_xlabel(AXIS_LABEL["lr"])
    axs[1].set_ylabel(AXIS_LABEL["beta"])
    axs[1].set_title("извлечённая граница (скелетон)")
    legend_fig(fig)
    save(fig, "f04_eta_beta.png")


def fig_theory():
    p = os.path.join(DATA_DIR, "H_theory.npz")
    if not os.path.exists(p):
        return
    z = np.load(p)
    xs, bs = z["xs"], z["bs"]
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.2))
    rho = z["rho"]
    im = axs[0].pcolormesh(xs, bs, np.minimum(rho, 2.0), cmap="viridis",
                           shading="auto", rasterized=True)
    axs[0].contour(xs, bs, rho, levels=[1.0], colors="w", linewidths=1.5)
    th = 2 * (1 + bs)
    axs[0].plot(th, bs, "r--", lw=1.5, label=r"$\eta\lambda = 2(1+\beta)$")
    axs[0].set_xlim(xs.min(), xs.max())
    axs[0].set_xlabel(r"$x=\eta\lambda$")
    axs[0].set_ylabel(r"$\beta$")
    axs[0].set_title(r"спектральный радиус $\rho(M)$")
    axs[0].legend(fontsize=7, loc="upper left")
    fig.colorbar(im, ax=axs[0], fraction=.046)

    axs[1].pcolormesh(xs, bs, z["stable_num"], cmap="Blues", shading="auto",
                      rasterized=True)
    axs[1].plot(th, bs, "r--", lw=1.5, label=r"теория: $2(1+\beta)$")
    axs[1].axvline(2.0, color="k", ls=":", lw=1.2, label=r"GD: $\eta\lambda=2$")
    axs[1].set_xlim(xs.min(), xs.max())
    axs[1].set_xlabel(r"$x=\eta\lambda$")
    axs[1].set_ylabel(r"$\beta$")
    axs[1].set_title("численная область устойчивости")
    axs[1].legend(fontsize=7, loc="upper left")
    save(fig, "f05_theory_stability.png")


def fig_bifurcation():
    p = os.path.join(DATA_DIR, "H_theory.npz")
    if not os.path.exists(p):
        return
    z = np.load(p)
    fig, axs = plt.subplots(1, 3, figsize=(7.6, 2.7), sharey=True)
    for ax, (e, t, b) in zip(axs, [(z["bif_eta0"], z["bif_traj0"], 0.0),
                                   (z["bif_eta5"], z["bif_traj5"], 0.5),
                                   (z["bif_eta9"], z["bif_traj9"], 0.9)]):
        tt = np.clip(t, -3, 3)
        E = np.repeat(e[None, :], tt.shape[0], axis=0)
        ax.plot(E.ravel(), tt.ravel(), ",", color="#1b4f9c", alpha=.5)
        ax.set_title(rf"$\beta={b}$")
        ax.set_xlabel(r"$\eta$")
        ax.set_ylim(-2.2, 2.2)
    axs[0].set_ylabel(r"$w_t$ (аттрактор)")
    fig.suptitle("Бифуркационная диаграмма: $L(w)=\\frac{1}{2}(w^2-y)^2$, GD с инерцией",
                 y=1.04, fontsize=9)
    save(fig, "f06_bifurcation.png")


def _panel_fig(names, titles, fname, ncols=None, figw=7.2, rowh=3.1):
    names = [(n, t) for n, t in zip(names, titles) if has(n)]
    if not names:
        return
    ncols = ncols or len(names)
    nrows = int(np.ceil(len(names) / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(figw, rowh * nrows))
    axs = np.atleast_1d(axs).ravel()
    for ax, (n, t) in zip(axs, names):
        m = meta(n)
        plot_regime(ax, n, f"{t}\n$D={m['D']:.3f}\\pm{m['sigma']:.3f}$")
    for ax in axs[len(names):]:
        ax.axis("off")
    legend_fig(fig)
    save(fig, fname)


def fig_wd():
    _panel_fig(["B2_eta_wd"], [r"сечение $(\eta,\lambda_{wd})$"],
               "f07_eta_wd.png", figw=3.8)


def fig_batch_strip():
    if not has("B3_eta_batch"):
        return
    r = load_result("B3_eta_batch")
    m = meta("B3_eta_batch")
    fig, ax = plt.subplots(figsize=(7.0, 2.4))
    ax.pcolormesh(r["xs"], r["ys"], r["regime"], cmap=REG_CMAP, norm=REG_NORM,
                  shading="auto", rasterized=True)
    ax.set_xscale("log")
    ax.set_xlabel(AXIS_LABEL["lr"])
    ax.set_ylabel(AXIS_LABEL["batch"])
    ax.set_title(r"сечение $(\eta,B)$, $|D|=32$")
    legend_fig(fig)
    save(fig, "f08_eta_batch.png")


def fig_minibatch():
    _panel_fig(["B4_mb_B1", "B4_mb_B4", "B4_mb_B32"],
               ["$B=1$", "$B=4$", "$B=32$ (полный батч)"],
               "f09_minibatch.png", ncols=3)


def fig_optimizers():
    _panel_fig(["C1_opt_sgd", "C2_opt_momentum", "C3_opt_adam", "C4_opt_rmsprop"],
               ["SGD", r"SGD+momentum, $\beta=0.9$", "Adam", "RMSprop"],
               "f10_optimizers.png", ncols=2, rowh=3.1)


def fig_adam_sections():
    _panel_fig(["C5_adam_eta_b1", "C6_adam_eta_b2"],
               [r"Adam: $(\eta,\beta_1)$", r"Adam: $(\eta,\beta_2)$"],
               "f11_adam_sections.png", ncols=2)


def fig_data():
    _panel_fig(["D1_synth16", "D2_digits16"],
               ["случайные данные, $|D|=16$", "digits (PCA-16), $|D|=16$"],
               "f12_data.png", ncols=2)


def fig_wd_slices():
    _panel_fig([f"E{i}_eta_beta_wd{i}" for i in range(4)],
               [r"$\lambda_{wd}=0$", r"$\lambda_{wd}=10^{-4}$",
                r"$\lambda_{wd}=10^{-3}$", r"$\lambda_{wd}=10^{-2}$"],
               "f13_wd_slices.png", ncols=2)


def fig_zoom():
    names = [f"F_zoom{k}" for k in range(5)]
    _panel_fig(names, [f"уровень зума {k}" for k in range(5)],
               "f14_zoom.png", ncols=3, rowh=2.9)


def fig_precision():
    _panel_fig(["E0_eta_beta_wd0", "G1_float32"],
               ["float64", "float32"], "f15_precision.png", ncols=2)


def fig_boundary_zoom_detail():
    """Крупный план границы для иллюстрации самоподобия."""
    if not (has("F_zoom0") and has("F_zoom4")):
        return
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.2))
    for ax, n, t in zip(axs, ["F_zoom0", "F_zoom4"],
                        ["уровень 0", "уровень 4 ($\\times 16$)"]):
        r = load_result(n)
        bnd = extract_boundary(r["regime"])
        ax.pcolormesh(r["xs"], r["ys"], bnd, cmap="binary", shading="auto",
                      rasterized=True)
        ax.set_xscale("log")
        ax.set_xlabel(AXIS_LABEL["lr"])
        ax.set_ylabel(AXIS_LABEL["beta"])
        ax.set_title(f"граница, {t}")
    save(fig, "f16_boundary_selfsim.png")


ALL = [fig_base_tanh, fig_activations, fig_boxcount, fig_eta_beta, fig_theory,
       fig_bifurcation, fig_wd, fig_batch_strip, fig_minibatch, fig_optimizers,
       fig_adam_sections, fig_data, fig_wd_slices, fig_zoom, fig_precision,
       fig_boundary_zoom_detail]

if __name__ == "__main__":
    for f in ALL:
        try:
            f()
        except Exception as e:  # noqa
            print(f"[warn] {f.__name__}: {e}")
