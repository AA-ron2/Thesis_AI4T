from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import NamedTuple

from procs.gym.helpers.fast_rollout import fast_simulate

"""
Calibration of Avellaneda-Stoikov parameters (A, κ, σ) from
Tardis L2 order book snapshot data.

Methodology follows the hftbacktest GLFT tutorial:
    https://hftbacktest.readthedocs.io/en/py-v2.1.0/tutorials/GLFT%20Market%20Making%20Model%20and%20Grid%20Trading.html

Three parameters are calibrated:

    σ  — midprice volatility (arithmetic, A-S convention)
    A  — baseline arrival rate  (trades per second at zero depth)
    κ  — fill probability decay (USDT⁻¹; same as 'kappa' / 'k' in A-S)

The fill probability model is:
    λ(δ) = A · exp(−κ · δ)

where δ is the depth of the limit order from the current mid-price
in price units (not ticks).

Arrival detection from L2 snapshots
------------------------------------
Tardis L2 data has no explicit trade column, but market order arrivals
are visible as changes in the best bid/ask between consecutive snapshots:

    Buy market order:  asks[0].price at t+1  >  asks[0].price at t
                       → a trade consumed the best ask at depth
                         δ = asks[0].price_t − mid_t

    Sell market order: bids[0].price at t+1  <  bids[0].price at t
                       → a trade consumed the best bid at depth
                         δ = mid_t − bids[0].price_t

This is identical to what hftbacktest's measure_trading_intensity records
during backtesting, adapted here for the pandas-based Tardis format.

Linear regression for A and κ
------------------------------
Taking logs of λ(δ) = A·exp(−κ·δ):
    log λ(δ) = log A − κ·δ

This is a standard OLS regression with slope −κ and intercept log A,
fitted on the near-mid depth range (δ ∈ [0, fit_depth_max]) where the
exponential model is most accurate (hftbacktest tutorial §"Calibrate A and k").

References
----------
Avellaneda & Stoikov (2008) — original model
Guéant, Lehalle & Fernandez-Tapia (2013) — GLFT market making
hftbacktest tutorial — calibration methodology
source — https://quant.stackexchange.com/questions/36073/how-does-one-calibrate-lambda-in-a-avellaneda-stoikov-market-making-problem
"""

class ASParameters(NamedTuple):
    """Calibrated Avellaneda-Stoikov parameters."""
    sigma: float    # midprice volatility (price units / sqrt(second))
    A: float        # arrival rate at zero depth (trades / second)
    kappa: float    # fill probability decay (1 / price unit)


def calibrate_as_parameters(
    filepath: str,
    tick_size: float = 0.00001,
    fit_depth_max: float | None = None,
    volatility_window_sec: float = 600.0,
    min_arrivals: int = 20,
    plot: bool = True,
    verbose: bool = True,
) -> ASParameters:
    """
    Calibrate σ, A, κ from a single day of Tardis L2 snapshot data.

    Parameters
    ----------
    filepath : str
        Path to a Tardis CSV file (binance_book_snapshot_25_<date>_<pair>.csv).
        Must have columns: timestamp, asks[0].price, bids[0].price.
    tick_size : float
        Minimum price increment for the asset. DOGE = 0.00001.
    fit_depth_max : float | None
        Maximum depth (in price units) to include in the regression.
        The near-mid range is most reliable; depths far from mid
        accumulate fewer trades and fit less well.
        Default: 5 * tick_size (i.e., first 5 tick levels).
    volatility_window_sec : float
        Rolling window in seconds for σ estimation. The final σ is the
        median of per-window estimates, which is robust to intraday
        volatility spikes. Default: 600 (10 minutes), matching hftbacktest.
    min_arrivals : int
        Minimum number of detected market orders at a given depth bin
        for that bin to be included in the regression. Bins with fewer
        observations produce noisy λ estimates.
    plot : bool
        If True, plots the empirical λ(δ) vs the fitted curve.
    verbose : bool
        If True, prints calibration summary.

    Returns
    -------
    ASParameters
        Named tuple with fields: sigma, A, kappa.

    Example
    -------
    >>> params = calibrate_as_parameters(
    ...     "datasets/binance_book_snapshot_25_2025-01-01_DOGEUSDT.csv",
    ...     tick_size=0.00001,
    ... )
    >>> print(f"σ={params.sigma:.6f}, A={params.A:.4f}, κ={params.kappa:.0f}")
    σ=0.000021, A=0.7834, κ=34821
    """
    # ── 1. Load data ──────────────────────────────────────────────────────────
    data = pd.read_csv(filepath, index_col="timestamp")
    data.index = pd.to_datetime(data.index, unit="us")

    best_ask = data["asks[0].price"].to_numpy(dtype=np.float64)
    best_bid = data["bids[0].price"].to_numpy(dtype=np.float64)
    mid      = (best_ask + best_bid) / 2.0

    ts_ns = data.index.view("int64")
    t_sec = (ts_ns - ts_ns[0]) / 1e9
    dt    = np.diff(t_sec, prepend=t_sec[0])   # Δt per snapshot (seconds)
    T     = t_sec[-1]
    N     = len(mid)

    if verbose:
        print(f"Loaded {N:,} snapshots over {T/3600:.2f} hours")

    # ── 2. Calibrate σ (arithmetic BM convention) ─────────────────────────────
    # A-S uses dS = σ dW, so σ has units of price / sqrt(second).
    # Compute: σ_window = std(ΔS) / sqrt(mean(Δt))  per rolling window.
    # Take the median across windows to be robust to volatility spikes.
    #
    # Critical: use absolute price differences, NOT log returns.
    # Log returns overcount σ by 1/S² ≈ factor of ~10 for DOGE at $0.32.
    dS = np.diff(mid)        # shape (N-1,)
    dt_mid = dt[1:]          # matching Δt values

    # Snapshots per window
    window_snaps = max(
        1,
        int(volatility_window_sec / np.median(dt_mid[dt_mid > 0]))
    )

    sigma_estimates = []
    for start in range(0, N - 1 - window_snaps, window_snaps):
        dS_w  = dS[start : start + window_snaps]
        dt_w  = dt_mid[start : start + window_snaps]
        total_t = dt_w.sum()
        if total_t > 0:
            # σ = sqrt(Var(ΔS) / E[Δt]) — scales to per-second units
            sigma_estimates.append(np.sqrt(np.sum(dS_w**2) / total_t))

    sigma = float(np.median(sigma_estimates))

    # ── 3. Detect market order arrivals ───────────────────────────────────────
    # A buy market order is visible when the best ask price increases
    # between snapshots (asks were consumed). Depth = ask_before − mid_before.
    # A sell market order: best bid decreases. Depth = mid_before − bid_before.
    #
    # We only count events where the best price *actually moved*, which
    # filters out passive quote updates that don't represent trades.

    arrival_depths = []   # depth from mid, in price units

    ask_prev = best_ask[:-1]
    ask_next = best_ask[1:]
    bid_prev = best_bid[:-1]
    bid_next = best_bid[1:]
    mid_prev = mid[:-1]

    # Buy market orders: best ask moved up → trade at ask
    buy_mask = ask_next > ask_prev + tick_size * 0.5
    buy_depths = ask_prev[buy_mask] - mid_prev[buy_mask]
    arrival_depths.extend(buy_depths[buy_depths >= 0].tolist())

    # Sell market orders: best bid moved down → trade at bid
    sell_mask = bid_next < bid_prev - tick_size * 0.5
    sell_depths = mid_prev[sell_mask] - bid_prev[sell_mask]
    arrival_depths.extend(sell_depths[sell_depths >= 0].tolist())

    arrival_depths = np.array(arrival_depths)
    n_arrivals = len(arrival_depths)

    if verbose:
        print(f"Detected {n_arrivals:,} market order arrivals "
              f"({n_arrivals / T:.2f}/sec average)")

    # ── 4. Build λ(δ) histogram ───────────────────────────────────────────────
    # Bin arrivals by depth. Each bin represents one tick width.
    # λ(δ) = count in bin / T (arrivals per second at that depth).
    #
    # hftbacktest uses half-tick bin centres: δ = 0.5*tick, 1.5*tick, ...
    # which avoids ambiguity at bin edges.

    if fit_depth_max is None:
        fit_depth_max = 5 * tick_size

    n_bins = max(1, int(fit_depth_max / tick_size))
    bin_edges   = np.arange(0, (n_bins + 1)) * tick_size
    bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2   # half-tick centres

    counts, _ = np.histogram(arrival_depths, bins=bin_edges)
    lambda_empirical = counts / T    # arrivals per second

    # ── 5. Fit A and κ via log-linear regression ──────────────────────────────
    # log λ(δ) = log A − κ·δ
    # slope = −κ,  intercept = log A
    # Only include bins with enough observations for a stable estimate.

    valid = counts >= min_arrivals
    if valid.sum() < 2:
        raise ValueError(
            f"Only {valid.sum()} bins have >= {min_arrivals} arrivals. "
            f"Try reducing fit_depth_max or min_arrivals, or use more data."
        )

    x = bin_centres[valid]         # depth values
    y = np.log(lambda_empirical[valid])   # log arrival rates

    # OLS (identical to hftbacktest's linear_regression @njit function)
    n   = len(x)
    sx  = x.sum();  sy  = y.sum()
    sx2 = (x**2).sum();  sxy = (x * y).sum()
    slope     = (n * sxy - sx * sy) / (n * sx2 - sx**2)
    intercept = (sy - slope * sx) / n

    kappa = float(-slope)
    A     = float(np.exp(intercept))

    # ── 6. Diagnostic plot ────────────────────────────────────────────────────
    if plot:
        delta_plot = np.linspace(0, bin_edges[-1], 200)
        lambda_fit = A * np.exp(-kappa * delta_plot)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.scatter(bin_centres * 1e5, lambda_empirical,
                   label="Empirical λ(δ)", color="steelblue", s=40, zorder=3)
        ax.plot(delta_plot * 1e5, lambda_fit,
                label=f"Fit: A={A:.3f}·exp(−{kappa:.0f}·δ)",
                color="crimson", linewidth=2)
        ax.axvline(fit_depth_max * 1e5, color="gray", linestyle="--",
                   alpha=0.6, label=f"Fit range ({n_bins} ticks)")
        ax.set_xlabel("Depth from mid (ticks × 10⁻⁵ = price units)")
        ax.set_ylabel("Arrival rate λ(δ)  [trades / second]")
        ax.set_title("Trading Intensity Calibration")
        ax.legend()
        ax.grid(True, alpha=0.4)
        plt.tight_layout()
        plt.show()

    # ── 7. Summary ────────────────────────────────────────────────────────────
    if verbose:
        print(f"\n── Calibrated A-S Parameters ──────────────────────")
        print(f"  σ     = {sigma:.6f}  (price / √s)")
        print(f"  A     = {A:.4f}   (trades / s at zero depth)")
        print(f"  κ     = {kappa:.0f}  (1 / price unit)")
        print(f"  Fit range: δ ∈ [0, {fit_depth_max:.5f}] ({n_bins} tick levels)")
        print(f"  Bins used in regression: {valid.sum()} / {n_bins}")

    return ASParameters(sigma=sigma, A=A, kappa=kappa)


"""
A-S parameter calibration via Optuna.

Tunes the risk-aversion parameter γ to maximise Sharpe ratio on
the training data.  κ and A are estimated from data (not tuned).

Usage::

    from procs.gym.calibration import tune_gamma
    best_gamma, study = tune_gamma(
        midprices=S, dt_array=dt_sec,
        sigma=sigma, kappa=35000, A=0.8,
        n_trials=100,
    )

Reference: Bergstra & Bengio (2012), Akiba et al. (2019, Optuna).
"""

def tune_gamma(
    midprices: np.ndarray,
    dt_array: np.ndarray,
    sigma: float,
    kappa: float,
    A: float,
    tick_size: float = 0.00001,
    Q_MAX: int = 10,
    gamma_range: tuple[float, float] = (0.001, 1.0),
    n_trials: int = 100,
    num_trajectories: int = 50,
    seed: int = 42,
    metric: str = "sharpe",
    verbose: bool = True,
):
    """
    Tune γ (risk aversion) via Optuna TPE to maximise Sharpe.

    Parameters
    ----------
    midprices, dt_array : np.ndarray
        Training day data.
    sigma, kappa, A : float
        Estimated model parameters (fixed during tuning).
    gamma_range : tuple
        Search range for γ (log-uniform).
    n_trials : int
        Number of Optuna trials.
    num_trajectories : int
        N for averaging (N=50 is a good speed/accuracy tradeoff).
    metric : str
        'sharpe' or 'sortino' — objective to maximise.

    Returns
    -------
    best_gamma : float
    study : optuna.Study
    """
    try:
        import optuna
    except ImportError:
        raise ImportError("optuna required. pip install optuna")

    T = float(dt_array.sum())

    def objective(trial):
        gamma = trial.suggest_float("gamma", gamma_range[0], gamma_range[1], log=True)
        stats = fast_simulate(
            midprices=midprices, dt_array=dt_array,
            gamma=gamma, sigma=sigma, kappa=kappa, A=A,
            terminal_time=T, tick_size=tick_size, Q_MAX=Q_MAX,
            num_trajectories=num_trajectories, seed=seed,
            use_linear_approximation=False,
        )
        return float(stats[metric].mean())

    if not verbose:
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    best_gamma = study.best_params["gamma"]
    if verbose:
        print(f"\nBest γ = {best_gamma:.6f}")
        print(f"Best {metric} = {study.best_value:.6f}")

    return best_gamma, study
