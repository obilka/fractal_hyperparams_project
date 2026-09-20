"""
fractal_hp.py
=============
Многомерные фрактальные границы обучаемости нейросетей:
шаг обучения, инерция (momentum), затухание весов (weight decay),
размер батча и адаптивные оптимизаторы (Adam, RMSprop).

Продолжение НИР "Фрактальная граница обучаемости нейросетей".

Основная идея реализации: все точки сетки гиперпараметров обучаются
одновременно как батч независимых сетей (vectorized ensemble). Это даёт
ускорение на 2-3 порядка по сравнению с поэлементным циклом и позволяет
сканировать сетки 256x256 / 512x512 на одном ядре CPU.

Автор: Бухаров Д. И., группа 25.Б21
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, asdict, field

import numpy as np

# --------------------------------------------------------------------------
# 0. Глобальные настройки воспроизводимости
# --------------------------------------------------------------------------

SEED = 42
DTYPE = np.float64
OUT_DIR = os.environ.get("FHP_OUT", os.path.join(os.path.dirname(__file__), "..", "outputs"))
DATA_DIR = os.path.join(OUT_DIR, "data")
FIG_DIR = os.path.join(OUT_DIR, "figures")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# Коды режимов
CONVERGED = 0
OSCILLATING = 1
DIVERGED = 2
NONFINITE = 3
REGIME_NAMES = {
    CONVERGED: "converged",
    OSCILLATING: "oscillating/periodic",
    DIVERGED: "diverged",
    NONFINITE: "NaN/Inf",
}

# Пороги критериев (см. раздел "Постановка задачи" отчёта)
DIV_THRESHOLD = 1e6      # порог расходимости для нормализованного loss
CONV_TOL = 1e-3          # относительный разброс на хвосте => стабилизация
TAIL = 64                # длина хвоста траектории, хранимой в памяти
TAIL_SHORT = 20          # окно усреднения для критерия сходимости


# --------------------------------------------------------------------------
# 1. Данные
# --------------------------------------------------------------------------

@dataclass
class DataSpec:
    kind: str = "synthetic"   # 'synthetic' | 'digits'
    n_samples: int = 8
    d_in: int = 16
    d_out: int = 16
    seed: int = SEED


def make_data(spec: DataSpec):
    """Возвращает (X, Y) в float64."""
    if spec.kind == "synthetic":
        rng = np.random.default_rng(spec.seed)
        X = rng.standard_normal((spec.n_samples, spec.d_in)).astype(DTYPE)
        Y = rng.standard_normal((spec.n_samples, spec.d_out)).astype(DTYPE)
        return X, Y
    if spec.kind == "digits":
        from sklearn.datasets import load_digits
        d = load_digits()
        rng = np.random.default_rng(spec.seed)
        idx = rng.permutation(len(d.data))[: spec.n_samples]
        Xr = d.data[idx].astype(DTYPE) / 16.0
        yr = d.target[idx]
        # центрирование + проекция на первые d_in главных компонент
        full = d.data.astype(DTYPE) / 16.0
        mu = full.mean(axis=0)
        U, S, Vt = np.linalg.svd(full - mu, full_matrices=False)
        P = Vt[: spec.d_in].T                       # (64, d_in)
        X = (Xr - mu) @ P
        X /= (X.std() + 1e-12)                      # нормировка масштаба
        Y = -np.ones((spec.n_samples, spec.d_out), dtype=DTYPE)
        Y[np.arange(spec.n_samples), yr % spec.d_out] = 1.0
        return X.astype(DTYPE), Y
    raise ValueError(f"unknown data kind: {spec.kind}")


# --------------------------------------------------------------------------
# 2. Конфигурация эксперимента
# --------------------------------------------------------------------------

@dataclass
class ScanConfig:
    """Описание одного двумерного сканирования пространства гиперпараметров."""
    name: str
    x_name: str                       # имя гиперпараметра по оси X
    y_name: str                       # имя гиперпараметра по оси Y
    x_range: tuple = (1e-3, 1e1)
    y_range: tuple = (1e-3, 1e1)
    x_log: bool = True
    y_log: bool = True
    resolution: int = 256
    resolution_y: int = 0             # 0 => совпадает с resolution
    n_iter: int = 300
    activation: str = "tanh"          # 'tanh' | 'relu' | 'linear'
    optimizer: str = "sgd"            # 'sgd' | 'momentum' | 'adam' | 'rmsprop'
    hidden: int = 16
    # фиксированные значения гиперпараметров (переопределяются осями сканирования)
    lr0: float = 0.1
    lr1: float = 0.1
    beta: float = 0.0                 # momentum
    wd: float = 0.0                   # weight decay (L2)
    batch: int = 0                    # 0 => полный батч
    beta1: float = 0.9                # Adam
    beta2: float = 0.999              # Adam / RMSprop
    eps: float = 1e-8
    data: DataSpec = field(default_factory=DataSpec)
    dtype: str = "float64"
    init_scale: float = 0.1
    chunk: int = 4096


def _axis(lo, hi, n, log):
    if log:
        return np.logspace(np.log10(lo), np.log10(hi), n)
    return np.linspace(lo, hi, n)


# --------------------------------------------------------------------------
# 3. Векторизованное обучение ансамбля сетей
# --------------------------------------------------------------------------

def _act(H, kind):
    if kind == "tanh":
        return np.tanh(H)
    if kind == "relu":
        return np.maximum(H, 0.0)
    return H


def _act_grad(H, Ha, kind):
    if kind == "tanh":
        return 1.0 - Ha * Ha
    if kind == "relu":
        return (H > 0.0).astype(Ha.dtype)
    return np.ones_like(Ha)


def _train_chunk(X, Y, params, cfg: ScanConfig, dt):
    """
    Обучает G независимых сетей с индивидуальными гиперпараметрами.

    params: dict массивов формы (G,) — lr0, lr1, beta, wd, batch, beta1, beta2.
    Возвращает dict со статистиками траекторий.
    """
    G = params["lr0"].shape[0]
    N, d_in = X.shape
    d_out = Y.shape[1]
    h = cfg.hidden

    rng = np.random.default_rng(cfg.data.seed + 1000)
    W0 = np.broadcast_to(
        (rng.standard_normal((1, h, d_in)) * cfg.init_scale).astype(dt), (G, h, d_in)
    ).copy()
    W1 = np.broadcast_to(
        (rng.standard_normal((1, d_out, h)) * cfg.init_scale).astype(dt), (G, d_out, h)
    ).copy()

    lr0 = params["lr0"].astype(dt)[:, None, None]
    lr1 = params["lr1"].astype(dt)[:, None, None]
    beta = params["beta"].astype(dt)[:, None, None]
    wd = params["wd"].astype(dt)[:, None, None]
    b1 = params["beta1"].astype(dt)[:, None, None]
    b2 = params["beta2"].astype(dt)[:, None, None]
    batch = params["batch"].astype(np.int64)

    opt = cfg.optimizer
    if opt in ("momentum", "adam", "rmsprop"):
        M0 = np.zeros_like(W0)
        M1 = np.zeros_like(W1)
    if opt in ("adam", "rmsprop"):
        V0 = np.zeros_like(W0)
        V1 = np.zeros_like(W1)

    # фиксированный порядок примеров для мини-батчей (общий для всех конфигураций)
    perm = np.random.default_rng(cfg.data.seed + 7).permutation(N)
    bs_chunk = int(batch[0])
    use_mb = bool(bs_chunk > 0 and bs_chunk < N)

    T = cfg.n_iter
    tail = np.zeros((G, TAIL), dtype=dt)
    sum_l = np.zeros(G, dtype=dt)
    sum_inv_l = np.zeros(G, dtype=dt)
    max_l = np.zeros(G, dtype=dt)
    bad = np.zeros(G, dtype=bool)
    L0 = None
    eps = cfg.eps

    for t in range(T):
        if use_mb:
            # мини-батч фиксированного размера; порядок примеров общий
            # для всех конфигураций чанка (детерминированный циклический обход)
            start = (t * bs_chunk) % N
            idx = perm[np.arange(start, start + bs_chunk) % N]
            Xb, Yb = X[idx], Y[idx]
        else:
            Xb, Yb = X, Y
        Nb = Xb.shape[0]

        H = np.einsum("nd,ghd->gnh", Xb, W0, optimize=True)
        Ha = _act(H, cfg.activation)
        out = np.einsum("gnh,goh->gno", Ha, W1, optimize=True)
        R = out - Yb
        if use_mb:
            # мониторинг режима ведётся по полной выборке, чтобы шум
            # мини-батча не искажал классификацию сходимости
            Hf = np.einsum("nd,ghd->gnh", X, W0, optimize=True)
            Rf = np.einsum("gnh,goh->gno", _act(Hf, cfg.activation), W1,
                           optimize=True) - Y
            loss = np.einsum("gno,gno->g", Rf, Rf) / (N * d_out)
            del Hf, Rf
        else:
            loss = np.einsum("gno,gno->g", R, R) / (Nb * d_out)

        if L0 is None:
            L0 = np.maximum(loss.copy(), 1e-300)
        ln = loss / L0

        nonfin = ~np.isfinite(ln)
        bad |= nonfin
        lnc = np.where(np.isfinite(ln), ln, DIV_THRESHOLD * 10.0)
        lnc = np.clip(lnc, 0.0, 1e12)
        sum_l += lnc
        sum_inv_l += 1.0 / np.maximum(lnc, 1e-10)
        max_l = np.maximum(max_l, lnc)
        tail[:, t % TAIL] = lnc

        # --- градиенты (ручное обратное распространение) ---
        go = (2.0 / (Nb * d_out)) * R
        gW1 = np.einsum("gno,gnh->goh", go, Ha, optimize=True)
        gha = np.einsum("gno,goh->gnh", go, W1, optimize=True)
        gh = gha * _act_grad(H, Ha, cfg.activation)
        gW0 = np.einsum("gnh,nd->ghd", gh, Xb, optimize=True)

        # L2-регуляризация (weight decay)
        gW0 = gW0 + wd * W0
        gW1 = gW1 + wd * W1

        # защита от переполнения: замораживаем «мёртвые» конфигурации
        np.nan_to_num(gW0, copy=False, nan=0.0, posinf=1e30, neginf=-1e30)
        np.nan_to_num(gW1, copy=False, nan=0.0, posinf=1e30, neginf=-1e30)

        if opt == "sgd":
            W0 -= lr0 * gW0
            W1 -= lr1 * gW1
        elif opt == "momentum":
            # heavy-ball: v_{t+1} = beta*v_t + g_t ; w_{t+1} = w_t - lr*v_{t+1}
            M0 = beta * M0 + gW0
            M1 = beta * M1 + gW1
            W0 -= lr0 * M0
            W1 -= lr1 * M1
        elif opt == "rmsprop":
            V0 = b2 * V0 + (1 - b2) * gW0 * gW0
            V1 = b2 * V1 + (1 - b2) * gW1 * gW1
            W0 -= lr0 * gW0 / (np.sqrt(V0) + eps)
            W1 -= lr1 * gW1 / (np.sqrt(V1) + eps)
        elif opt == "adam":
            M0 = b1 * M0 + (1 - b1) * gW0
            M1 = b1 * M1 + (1 - b1) * gW1
            V0 = b2 * V0 + (1 - b2) * gW0 * gW0
            V1 = b2 * V1 + (1 - b2) * gW1 * gW1
            bc1 = 1 - np.power(b1, t + 1)
            bc2 = 1 - np.power(b2, t + 1)
            W0 -= lr0 * (M0 / bc1) / (np.sqrt(V0 / bc2) + eps)
            W1 -= lr1 * (M1 / bc1) / (np.sqrt(V1 / bc2) + eps)
        else:
            raise ValueError(f"unknown optimizer {opt}")

        np.nan_to_num(W0, copy=False, nan=np.nan, posinf=1e150, neginf=-1e150)
        np.nan_to_num(W1, copy=False, nan=np.nan, posinf=1e150, neginf=-1e150)

    order = [(t % TAIL) for t in range(T - TAIL, T)] if T >= TAIL else list(range(T))
    tail_ord = tail[:, order]
    return dict(tail=tail_ord, sum_l=sum_l, sum_inv_l=sum_inv_l, max_l=max_l, bad=bad)


def classify(stats):
    """Классификация режимов по хвосту траектории нормализованного loss."""
    tail = stats["tail"]
    G = tail.shape[0]
    regime = np.full(G, OSCILLATING, dtype=np.int8)

    m20 = tail[:, -TAIL_SHORT:].mean(axis=1)
    m20_first = tail[:, :TAIL_SHORT].mean(axis=1)
    m_tail = tail.mean(axis=1)
    rel = tail.std(axis=1) / np.maximum(m_tail, 1e-300)

    stabilized = rel < CONV_TOL                     # вышли на стационар
    decreasing = m20 < 0.99 * m20_first             # loss продолжает убывать
    conv = (m20 < 1.0) & (stabilized | decreasing) & np.isfinite(m20)
    div = (stats["max_l"] > DIV_THRESHOLD) | (m20 > DIV_THRESHOLD)
    nonfin = stats["bad"]

    regime[conv] = CONVERGED
    regime[div & ~nonfin] = DIVERGED
    regime[nonfin] = NONFINITE
    return regime


def run_scan(cfg: ScanConfig, verbose=True):
    """Полное двумерное сканирование. Возвращает словарь массивов (res x res)."""
    t0 = time.time()
    dt = np.float32 if cfg.dtype == "float32" else np.float64
    X, Y = make_data(cfg.data)
    X = X.astype(dt)
    Y = Y.astype(dt)

    n = cfg.resolution
    ny = cfg.resolution_y or n
    xs = _axis(cfg.x_range[0], cfg.x_range[1], n, cfg.x_log)
    ys = _axis(cfg.y_range[0], cfg.y_range[1], ny, cfg.y_log)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")   # YY меняется по строкам

    base = dict(
        lr0=np.full(XX.size, cfg.lr0), lr1=np.full(XX.size, cfg.lr1),
        beta=np.full(XX.size, cfg.beta), wd=np.full(XX.size, cfg.wd),
        batch=np.full(XX.size, cfg.batch), beta1=np.full(XX.size, cfg.beta1),
        beta2=np.full(XX.size, cfg.beta2),
    )

    def apply_axis(name, values):
        v = values.ravel()
        if name == "lr":
            base["lr0"] = v
            base["lr1"] = v
        elif name in base:
            base[name] = v
        else:
            raise ValueError(f"unknown axis {name}")

    apply_axis(cfg.x_name, XX)
    apply_axis(cfg.y_name, YY)
    base["batch"] = np.rint(base["batch"]).astype(np.int64)

    G = XX.size
    ckpt = os.path.join(DATA_DIR, f"{cfg.name}.part.npz")
    if os.path.exists(ckpt):
        z = np.load(ckpt)
        regime, sum_l, sum_inv, done = (z["regime"].copy(), z["sum_l"].copy(),
                                        z["sum_inv"].copy(), z["done"].copy())
        if verbose:
            print(f"  [{cfg.name}] resume: {done.mean()*100:.1f}% готово", flush=True)
    else:
        regime = np.zeros(G, dtype=np.int8)
        sum_l = np.zeros(G)
        sum_inv = np.zeros(G)
        done = np.zeros(G, dtype=bool)
    # конфигурации группируются по размеру батча: внутри чанка он должен
    # быть одинаковым (векторизация по ансамблю сетей)
    groups = [np.where(base["batch"] == b)[0] for b in np.unique(base["batch"])]
    done_cnt = 0
    for gidx in groups:
        for s in range(0, gidx.size, cfg.chunk):
            sel = gidx[s:s + cfg.chunk]
            if done[sel].all():
                done_cnt += sel.size
                continue
            p = {k: np.ascontiguousarray(v[sel]) for k, v in base.items()}
            st = _train_chunk(X, Y, p, cfg, dt)
            regime[sel] = classify(st)
            sum_l[sel] = st["sum_l"]
            sum_inv[sel] = st["sum_inv_l"]
            done[sel] = True
            done_cnt += sel.size
            np.savez(ckpt, regime=regime, sum_l=sum_l, sum_inv=sum_inv, done=done)
            if verbose and (done_cnt // cfg.chunk) % 8 == 0:
                frac = done_cnt / G
                el = time.time() - t0
                print(f"  [{cfg.name}] {frac*100:5.1f}%  elapsed {el:6.1f}s  "
                      f"eta {el/max(frac,1e-9)*(1-frac):6.1f}s", flush=True)

    out = dict(
        regime=regime.reshape(ny, n),
        sum_l=sum_l.reshape(ny, n),
        sum_inv=sum_inv.reshape(ny, n),
        xs=xs, ys=ys,
    )
    out["time_sec"] = time.time() - t0
    if os.path.exists(ckpt):
        os.remove(ckpt)
    if verbose:
        frac = {REGIME_NAMES[k]: float((out["regime"] == k).mean()) for k in range(4)}
        print(f"  [{cfg.name}] done in {out['time_sec']:.1f}s  fractions={frac}", flush=True)
    return out


# --------------------------------------------------------------------------
# 4. Извлечение границы и box-counting
# --------------------------------------------------------------------------

def extract_boundary(regime):
    """
    Бинарная граница между областью сходимости и всем остальным.
    Морфологический градиент (dilation XOR erosion) + скелетонизация.
    """
    from scipy import ndimage
    from skimage.morphology import skeletonize

    B = (regime == CONVERGED)
    st = ndimage.generate_binary_structure(2, 1)
    d = ndimage.binary_dilation(B, structure=st)
    e = ndimage.binary_erosion(B, structure=st)
    bnd = np.logical_xor(d, e)
    if bnd.sum() > 0:
        bnd = skeletonize(bnd)
    return bnd


def box_counting_dimension(boundary, min_box=2, max_div=8):
    """
    Оценка box-counting размерности. Возвращает (D, sigma_D, sizes, counts).
    sigma_D — стандартная ошибка наклона линейной регрессии log N ~ log eps.
    """
    b = np.asarray(boundary, dtype=bool)
    if b.sum() < 16:
        return float("nan"), float("nan"), [], []
    n = min(b.shape)
    max_box = max(min_box * 2, n // max_div)
    sizes, counts = [], []
    s = min_box
    while s <= max_box:
        ny = (b.shape[0] // s) * s
        nx = (b.shape[1] // s) * s
        red = b[:ny, :nx].reshape(ny // s, s, nx // s, s).any(axis=(1, 3))
        c = int(red.sum())
        if c > 0:
            sizes.append(s)
            counts.append(c)
        s *= 2
    if len(sizes) < 3:
        return float("nan"), float("nan"), sizes, counts
    ls = np.log(np.array(sizes, dtype=float))
    lc = np.log(np.array(counts, dtype=float))
    coef, cov = np.polyfit(ls, lc, 1, cov=True)
    D = -coef[0]
    sD = float(np.sqrt(cov[0, 0]))
    return float(D), sD, sizes, counts


def analyze(result):
    """Извлечение границы + размерность для результата сканирования."""
    bnd = extract_boundary(result["regime"])
    D, sD, sizes, counts = box_counting_dimension(bnd)
    return dict(boundary=bnd, D=D, sigma=sD, box_sizes=sizes, box_counts=counts,
                conv_frac=float((result["regime"] == CONVERGED).mean()),
                osc_frac=float((result["regime"] == OSCILLATING).mean()),
                div_frac=float((result["regime"] == DIVERGED).mean()),
                nan_frac=float((result["regime"] == NONFINITE).mean()))


# --------------------------------------------------------------------------
# 5. Теоретический блок: устойчивость GD с инерцией на квадратике
# --------------------------------------------------------------------------

def momentum_spectral_radius(eta_lambda, beta):
    """
    Спектральный радиус матрицы перехода
        M = [[1 - eta*lam, -eta*beta], [lam, beta]]
    для L(w) = 0.5*lam*w^2, v_{t+1} = beta v_t + lam w_t, w_{t+1} = w_t - eta v_{t+1}.
    Записан через безразмерный параметр x = eta*lam.
    """
    x = np.asarray(eta_lambda, dtype=float)
    b = np.asarray(beta, dtype=float)
    tr = 1.0 + b - x
    det = b
    disc = tr * tr - 4.0 * det
    r = np.empty(np.broadcast(x, b).shape, dtype=float)
    real = disc >= 0
    sq = np.sqrt(np.abs(disc))
    r_real = np.maximum(np.abs((tr + sq) / 2.0), np.abs((tr - sq) / 2.0))
    r_cplx = np.sqrt(np.abs(det))
    r = np.where(real, r_real, np.broadcast_to(r_cplx, r.shape))
    return r


def momentum_quadratic_experiment(n_eta=400, n_beta=400, T=400, lam=1.0):
    """Численная проверка области устойчивости в плоскости (eta*lam, beta)."""
    xs = np.linspace(1e-3, 5.0, n_eta)          # x = eta * lam
    bs = np.linspace(0.0, 0.99, n_beta)
    XX, BB = np.meshgrid(xs, bs)
    w = np.ones_like(XX)
    v = np.zeros_like(XX)
    eta = XX / lam
    for _ in range(T):
        v = BB * v + lam * w
        w = w - eta * v
        w = np.clip(np.nan_to_num(w, nan=1e30, posinf=1e30, neginf=-1e30), -1e30, 1e30)
        v = np.clip(np.nan_to_num(v, nan=1e30, posinf=1e30, neginf=-1e30), -1e30, 1e30)
    stable_num = np.abs(w) < 1e-3
    rho = momentum_spectral_radius(XX, BB)
    return dict(xs=xs, bs=bs, stable_num=stable_num, rho=rho, XX=XX, BB=BB)


# --------------------------------------------------------------------------
# 6. Сохранение
# --------------------------------------------------------------------------

def save_result(name, result, extra=None):
    path = os.path.join(DATA_DIR, f"{name}.npz")
    np.savez_compressed(
        path,
        regime=result["regime"].astype(np.int8),
        sum_l=result["sum_l"].astype(np.float32),
        sum_inv=result["sum_inv"].astype(np.float32),
        xs=result["xs"], ys=result["ys"],
    )
    if extra is not None:
        with open(os.path.join(DATA_DIR, f"{name}.json"), "w", encoding="utf-8") as f:
            json.dump(extra, f, ensure_ascii=False, indent=2)
    return path


def load_result(name):
    z = np.load(os.path.join(DATA_DIR, f"{name}.npz"))
    return dict(regime=z["regime"], sum_l=z["sum_l"], sum_inv=z["sum_inv"],
                xs=z["xs"], ys=z["ys"])
