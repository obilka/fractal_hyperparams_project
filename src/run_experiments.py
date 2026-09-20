"""
run_experiments.py — полный набор экспериментов НИР.

Запуск:  python3 run_experiments.py            (все стадии, resume-режим)
         python3 run_experiments.py A B C      (только выбранные стадии)

Результаты сохраняются в outputs/data/<name>.npz (карты режимов) и
outputs/data/<name>.json (метаданные + box-counting размерность).
Повторный запуск пропускает уже посчитанные эксперименты.
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings

import numpy as np

from fractal_hp import (ScanConfig, DataSpec, run_scan, analyze, save_result,
                        load_result, DATA_DIR, momentum_quadratic_experiment,
                        momentum_spectral_radius, extract_boundary,
                        box_counting_dimension, CONVERGED)

warnings.filterwarnings("ignore")

# ------------------------- общие параметры -------------------------------
T_ITER = 400
INIT = 1.0
RES_HI = 512
RES = 256
ETA1 = (1e-2, 1e1)      # базовый диапазон шага обучения (|D| = 1)
ETA_MB = (1e-1, 1e3)    # диапазон для датасетов большего размера
D1 = DataSpec(kind="synthetic", n_samples=1, d_in=16, d_out=16, seed=42)
D32 = DataSpec(kind="synthetic", n_samples=32, d_in=16, d_out=16, seed=42)
D16 = DataSpec(kind="synthetic", n_samples=16, d_in=16, d_out=16, seed=42)
DIG16 = DataSpec(kind="digits", n_samples=16, d_in=16, d_out=16, seed=42)


def base(**kw):
    kw.setdefault("n_iter", T_ITER)
    kw.setdefault("init_scale", INIT)
    kw.setdefault("resolution", RES)
    kw.setdefault("data", D1)
    kw.setdefault("x_range", ETA1)
    kw.setdefault("y_range", ETA1)
    return ScanConfig(**kw)


def run(cfg: ScanConfig, note=""):
    """Считает эксперимент, если он ещё не посчитан; возвращает (result, meta)."""
    npz = os.path.join(DATA_DIR, f"{cfg.name}.npz")
    js = os.path.join(DATA_DIR, f"{cfg.name}.json")
    if os.path.exists(npz) and os.path.exists(js):
        print(f"[skip] {cfg.name}", flush=True)
        return load_result(cfg.name), json.load(open(js, encoding="utf-8"))
    print(f"[run ] {cfg.name}  ({note})", flush=True)
    t0 = time.time()
    res = run_scan(cfg, verbose=True)
    a = analyze(res)
    meta = dict(
        name=cfg.name, note=note, D=a["D"], sigma=a["sigma"],
        box_sizes=a["box_sizes"], box_counts=a["box_counts"],
        conv_frac=a["conv_frac"], osc_frac=a["osc_frac"],
        div_frac=a["div_frac"], nan_frac=a["nan_frac"],
        seconds=time.time() - t0,
        config={k: (v if not hasattr(v, "__dict__") else v.__dict__)
                for k, v in cfg.__dict__.items()},
    )
    save_result(cfg.name, res, meta)
    print(f"[done] {cfg.name}: D={a['D']:.3f}+-{a['sigma']:.3f} "
          f"conv={a['conv_frac']:.2f} t={meta['seconds']:.0f}s", flush=True)
    return res, meta


# =========================== СТАДИИ =======================================

def stage_A():
    """Репликация базового эксперимента предыдущей НИР (активации)."""
    run(base(name="A1_base_tanh", x_name="lr0", y_name="lr1",
             activation="tanh", resolution=RES_HI), "базовая карта (eta0,eta1), tanh")
    run(base(name="A2_base_relu", x_name="lr0", y_name="lr1",
             activation="relu"), "(eta0,eta1), ReLU")
    run(base(name="A3_base_linear", x_name="lr0", y_name="lr1",
             activation="linear"), "(eta0,eta1), линейная сеть")
    # окна, центрированные на собственном пороге устойчивости активации
    run(base(name="A4_linear_wide", x_name="lr0", y_name="lr1",
             activation="linear", x_range=(1e-4, 1e-1), y_range=(1e-4, 1e-1)),
        "линейная сеть, собственное окно")
    run(base(name="A5_relu_mb", x_name="lr0", y_name="lr1", activation="relu",
             x_range=ETA_MB, y_range=ETA_MB, data=D32, batch=32),
        "ReLU, |D|=32, полный батч")
    run(base(name="A7_relu_mb_low", x_name="lr0", y_name="lr1", activation="relu",
             x_range=(1e-3, 1e1), y_range=(1e-3, 1e1), data=D32, batch=32),
        "ReLU, |D|=32, окно по своему порогу")
    run(base(name="A6_tanh_mb", x_name="lr0", y_name="lr1", activation="tanh",
             x_range=ETA_MB, y_range=ETA_MB, data=D32, batch=32),
        "tanh, |D|=32, полный батч")


def stage_B():
    """Новые двумерные сечения: (eta,beta), (eta,wd), (eta,B)."""
    run(base(name="B1_eta_beta", x_name="lr", y_name="beta",
             y_range=(0.0, 0.99), y_log=False, optimizer="momentum",
             resolution=RES_HI), "сечение (eta,beta), SGD+momentum")
    run(base(name="B2_eta_wd", x_name="lr", y_name="wd",
             y_range=(1e-6, 1e0), y_log=True), "сечение (eta,lambda_wd)")
    run(base(name="B3_eta_batch", x_name="lr", y_name="batch",
             x_range=ETA_MB, y_range=(1, 32), y_log=False,
             resolution=RES_HI, resolution_y=32, data=D32),
        "сечение (eta,B), |D|=32")
    for b in (1, 4, 32):
        run(base(name=f"B4_mb_B{b}", x_name="lr0", y_name="lr1",
                 x_range=ETA_MB, y_range=ETA_MB, data=D32, batch=b),
            f"(eta0,eta1), |D|=32, B={b}")


def stage_C():
    """Сравнение оптимизаторов."""
    run(base(name="C1_opt_sgd", x_name="lr0", y_name="lr1"), "SGD")
    run(base(name="C2_opt_momentum", x_name="lr0", y_name="lr1",
             x_range=(1e-3, 1e1), y_range=(1e-3, 1e1),
             optimizer="momentum", beta=0.9), "SGD+momentum, beta=0.9")
    run(base(name="C3_opt_adam", x_name="lr0", y_name="lr1",
             x_range=ETA_MB, y_range=ETA_MB, optimizer="adam"), "Adam")
    run(base(name="C4_opt_rmsprop", x_name="lr0", y_name="lr1",
             x_range=(1e-2, 1e2), y_range=(1e-2, 1e2),
             optimizer="rmsprop"), "RMSprop")
    run(base(name="C5_adam_eta_b1", x_name="lr", y_name="beta1",
             x_range=(1e-2, 1e2), y_range=(0.0, 0.999), y_log=False,
             optimizer="adam"), "Adam, сечение (eta,beta1)")
    run(base(name="C6_adam_eta_b2", x_name="lr", y_name="beta2",
             x_range=(1e-2, 1e2), y_range=(0.5, 0.9999), y_log=False,
             optimizer="adam"), "Adam, сечение (eta,beta2)")


def stage_D():
    """Синтетические против структурированных данных."""
    run(base(name="D1_synth16", x_name="lr0", y_name="lr1",
             x_range=ETA_MB, y_range=ETA_MB, data=D16, batch=16),
        "случайные данные, |D|=16")
    run(base(name="D2_digits16", x_name="lr0", y_name="lr1",
             x_range=ETA_MB, y_range=ETA_MB, data=DIG16, batch=16),
        "sklearn digits (PCA-16), |D|=16")


def stage_E():
    """Трёхмерное пространство (eta,beta,lambda_wd): набор сечений."""
    for i, wd in enumerate([0.0, 1e-4, 1e-3, 1e-2]):
        run(base(name=f"E{i}_eta_beta_wd{i}", x_name="lr", y_name="beta",
                 y_range=(0.0, 0.99), y_log=False, optimizer="momentum",
                 wd=wd), f"сечение (eta,beta) при lambda_wd={wd:g}")


def stage_F():
    """Последовательность зумов в плоскости (eta,beta)."""
    from scipy import ndimage
    src = "B1_eta_beta"
    if not os.path.exists(os.path.join(DATA_DIR, f"{src}.npz")):
        print("[F] нет базовой карты B1 — пропуск", flush=True)
        return
    r = load_result(src)
    bnd = extract_boundary(r["regime"])
    dens = ndimage.uniform_filter(bnd.astype(float), size=41)
    m = np.zeros_like(dens, dtype=bool)
    m[64:-64, 64:-64] = True                     # центр вдали от краёв
    iy, ix = np.unravel_index(np.argmax(np.where(m, dens, -1)), dens.shape)
    xs, ys = r["xs"], r["ys"]
    cx = float(np.log10(xs[ix]))
    cy = float(ys[iy])
    w0 = (np.log10(xs[-1]) - np.log10(xs[0])) / 2.0
    w1 = (ys[-1] - ys[0]) / 2.0
    center = dict(log10_eta=cx, beta=cy)
    with open(os.path.join(DATA_DIR, "F_zoom_center.json"), "w") as f:
        json.dump(center, f, indent=2)
    print(f"[F] центр зума: eta={10**cx:.4g}, beta={cy:.4g}", flush=True)
    for k in range(5):
        hw0, hw1 = w0 / 2 ** k, w1 / 2 ** k
        xr = (10 ** (cx - hw0), 10 ** (cx + hw0))
        yr = (max(0.0, cy - hw1), min(0.999, cy + hw1))
        run(base(name=f"F_zoom{k}", x_name="lr", y_name="beta",
                 x_range=xr, y_range=yr, y_log=False,
                 optimizer="momentum"), f"зум уровня {k}")


def stage_G():
    """Влияние численной точности."""
    run(base(name="G1_float32", x_name="lr", y_name="beta",
             y_range=(0.0, 0.99), y_log=False, optimizer="momentum",
             dtype="float32"), "(eta,beta), float32")


def stage_H():
    """Теоретический блок: устойчивость momentum на квадратике + бифуркации."""
    path = os.path.join(DATA_DIR, "H_theory.npz")
    if os.path.exists(path):
        print("[skip] H_theory", flush=True)
        return
    print("[run ] H_theory", flush=True)
    th = momentum_quadratic_experiment(n_eta=400, n_beta=400, T=400, lam=1.0)

    # бифуркационная диаграмма для нелинейной игрушечной задачи
    # L(w) = 0.5*(w^2 - y)^2 ; GD с инерцией
    def bifurcation(beta, n_eta=600, T=600, keep=120, y=1.0):
        etas = np.linspace(0.01, 1.6, n_eta)
        w = np.full(n_eta, 0.3)
        v = np.zeros(n_eta)
        traj = np.zeros((keep, n_eta))
        for t in range(T):
            g = 2.0 * w * (w * w - y)
            v = beta * v + g
            w = w - etas * v
            w = np.clip(np.nan_to_num(w, nan=1e6, posinf=1e6, neginf=-1e6), -1e6, 1e6)
            v = np.clip(np.nan_to_num(v, nan=1e6, posinf=1e6, neginf=-1e6), -1e6, 1e6)
            if t >= T - keep:
                traj[t - (T - keep)] = w
        return etas, traj

    b0 = bifurcation(0.0)
    b5 = bifurcation(0.5)
    b9 = bifurcation(0.9)
    np.savez_compressed(
        path,
        xs=th["xs"], bs=th["bs"], stable_num=th["stable_num"], rho=th["rho"],
        bif_eta0=b0[0], bif_traj0=b0[1],
        bif_eta5=b5[0], bif_traj5=b5[1],
        bif_eta9=b9[0], bif_traj9=b9[1],
    )
    print("[done] H_theory", flush=True)


STAGES = dict(A=stage_A, B=stage_B, C=stage_C, D=stage_D,
              E=stage_E, F=stage_F, G=stage_G, H=stage_H)

if __name__ == "__main__":
    sel = sys.argv[1:] or list(STAGES)
    t0 = time.time()
    for s in sel:
        print(f"===== stage {s} =====", flush=True)
        STAGES[s]()
    print(f"ALL DONE in {(time.time()-t0)/60:.1f} min", flush=True)
