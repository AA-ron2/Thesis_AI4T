"""
run_one_day.py
--------------
Processes a single day of the A-S baseline.
Called by SLURM array jobs via:
    python run_one_day.py --day-index $SLURM_ARRAY_TASK_ID

Saves results to: results/day_YYYY-MM-DD.csv
"""

import argparse
import sys
import pathlib
import numpy as np
import pandas as pd
from datetime import datetime

# ── Project root ──────────────────────────────────────────────────────────────
PROJECT_ROOT = next(
    (p for p in [pathlib.Path.cwd(), *pathlib.Path.cwd().parents]
     if (p / "procs").exists()),
    pathlib.Path.cwd()
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from procs.gym.calibration import tune_gamma
from procs.gym.data_loader import load_multi_day
from procs.gym.experiment_config import ReplayExperimentConfig
from procs.gym.helpers.fast_rollout import fast_simulate


# ── Calibration helper (copied from notebook) ─────────────────────────────────
def calibrate_from_arrays(
    S: np.ndarray,
    dt: np.ndarray,
    tick_size: float = 0.00001,
    n_depth_ticks: int = 5,
    min_arrivals: int = 10,
) -> tuple[float, float, float]:
    T = float(dt.sum())
    N = len(S)

    dS = np.diff(S)
    dt_mid = dt[1:]
    window_size = max(1, int(600.0 / np.median(dt_mid[dt_mid > 0])))
    sigma_estimates = []
    for start in range(0, N - 1 - window_size, window_size):
        dS_w = dS[start:start + window_size]
        dt_w = dt_mid[start:start + window_size]
        total_t = dt_w.sum()
        if total_t > 0:
            sigma_estimates.append(np.sqrt(np.sum(dS_w ** 2) / total_t))
    sigma = float(np.median(sigma_estimates)) if sigma_estimates else float(
        np.sqrt(np.sum(dS ** 2) / T))

    mid_diff = np.abs(np.diff(S))
    arrival_mask = mid_diff >= tick_size * 0.5
    arrival_depths = mid_diff[arrival_mask]

    bin_edges = np.arange(0, n_depth_ticks + 1) * tick_size
    bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2
    counts, _ = np.histogram(arrival_depths, bins=bin_edges)
    lambda_emp = counts / T

    valid = counts >= min_arrivals
    if valid.sum() < 2:
        return sigma, float(len(arrival_depths) / T), 35_000.0

    x = bin_centres[valid]
    y = np.log(lambda_emp[valid])
    n = len(x)
    slope = (n * np.dot(x, y) - x.sum() * y.sum()) / \
            (n * np.dot(x, x) - x.sum() ** 2)
    intercept = (y.sum() - slope * x.sum()) / n
    kappa = float(-slope)
    A = float(np.exp(intercept))
    return sigma, A, kappa


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--day-index", type=int, required=True,
                        help="Index into the sorted list of available days")
    args = parser.parse_args()

    cfg = ReplayExperimentConfig()
    cfg.ensure_artifact_dirs()

    # Load all days, then select just the one we need
    daily_S, daily_dt, dates = load_multi_day(str(cfg.datasets_dir), pair=cfg.pair)
    n_days = len(dates)

    if args.day_index >= n_days:
        print(f"Day index {args.day_index} out of range (only {n_days} days available). Exiting.")
        sys.exit(0)

    S    = daily_S[args.day_index]
    dt   = daily_dt[args.day_index]
    date = dates[args.day_index]

    print(f"[{datetime.now()}] Processing day {args.day_index + 1}/{n_days}: {date}")

    T = float(dt.sum())
    sigma, A, kappa = calibrate_from_arrays(S, dt, tick_size=cfg.tick_size)
    print(f"  Calibrated: sigma={sigma:.6f}  A={A:.3f}  kappa={kappa:.0f}")

    as_gamma, _ = tune_gamma(
        midprices=S,
        dt_array=dt,
        sigma=sigma,
        kappa=kappa,
        A=A,
        tick_size=cfg.tick_size,
        Q_MAX=cfg.q_max,
        gamma_range=cfg.as_gamma_range,
        n_trials=cfg.as_gamma_trials,
        num_trajectories=cfg.evaluation_rollouts,
        seed=cfg.evaluation_seed,
        verbose=False,
    )
    print(f"  Tuned gamma={as_gamma:.4f}")

    stats = fast_simulate(
        midprices=S,
        dt_array=dt,
        gamma=as_gamma,
        sigma=sigma,
        kappa=kappa,
        A=A,
        terminal_time=T,
        tick_size=cfg.tick_size,
        Q_MAX=cfg.q_max,
        num_trajectories=cfg.evaluation_rollouts,
        seed=cfg.evaluation_seed,
        use_linear_approximation=False,
    )

    row = {
        "Day": date,
        "Sharpe": float(stats["sharpe"].mean()),
        "Sortino": float(stats["sortino"].mean()),
        "Max DD": float(stats["max_drawdown"].mean()),
        "P&L-to-MAP": float(stats["pnl_to_map"].mean()),
        "Final PnL": float(stats["total_pnl"].mean()),
        "Mean |q|": float(stats["mean_abs_q"].mean()),
        "Near Cap Fraction": float(stats["near_cap_fraction"].mean()),
        "sigma": sigma,
        "A": A,
        "kappa": kappa,
        "as_gamma": as_gamma,
    }

    out_path = PROJECT_ROOT / "results" / f"day_{date}.csv"
    pd.DataFrame([row]).to_csv(out_path, index=False)
    print(f"[{datetime.now()}] Saved: {out_path}")
    print(
        f"  Sharpe={row['Sharpe']:+.4f}  MaxDD={row['Max DD']:.4f}  "
        f"PnL={row['Final PnL']:+.4f}"
    )


if __name__ == "__main__":
    main()
