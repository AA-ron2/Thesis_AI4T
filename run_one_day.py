"""
run_one_day.py
--------------
Processes a single day of the A-S baseline from a manifest row.
Called by SLURM array jobs via:
    python run_one_day.py --manifest results/baseline_manifest.csv --day-index $SLURM_ARRAY_TASK_ID
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from datetime import datetime

import pandas as pd

PROJECT_ROOT = next(
    (p for p in [pathlib.Path.cwd(), *pathlib.Path.cwd().parents] if (p / "procs").exists()),
    pathlib.Path.cwd(),
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from procs.gym.calibration import calibrate_from_arrays, tune_gamma
from procs.gym.data_loader import load_single_day
from procs.gym.experiment_config import ReplayExperimentConfig
from procs.gym.helpers.fast_rollout import fast_simulate

MANIFEST_COLUMNS = {"day_index", "date", "input_path", "result_path"}


def load_manifest_row(manifest_path: pathlib.Path, day_index: int) -> tuple[pd.Series, int]:
    manifest = pd.read_csv(manifest_path)
    missing_columns = MANIFEST_COLUMNS.difference(manifest.columns)
    if missing_columns:
        raise ValueError(
            f"Manifest missing required columns: {sorted(missing_columns)}"
        )

    row = manifest.loc[manifest["day_index"] == day_index]
    if row.empty:
        raise IndexError(f"Day index {day_index} not present in manifest {manifest_path}")
    if len(row) > 1:
        raise ValueError(f"Day index {day_index} appears multiple times in {manifest_path}")

    return row.iloc[0], len(manifest)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--day-index",
        type=int,
        required=True,
        help="Index into the baseline manifest day list",
    )
    parser.add_argument(
        "--manifest",
        type=pathlib.Path,
        required=True,
        help="CSV manifest written by baseline_snellius.py prepare",
    )
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    cfg = ReplayExperimentConfig()
    cfg.ensure_artifact_dirs()

    day_record, n_days = load_manifest_row(manifest_path, args.day_index)
    date = str(day_record["date"])
    input_path = pathlib.Path(str(day_record["input_path"])).resolve()
    out_path = pathlib.Path(str(day_record["result_path"])).resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found for day {date}: {input_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    midprices, dt_array, _ = load_single_day(str(input_path))

    print(f"[{datetime.now()}] Processing day {args.day_index + 1}/{n_days}: {date}")

    terminal_time = float(dt_array.sum())
    sigma, A, kappa = calibrate_from_arrays(
        midprices,
        dt_array,
        tick_size=cfg.tick_size,
    )
    print(f"  Calibrated: sigma={sigma:.6f}  A={A:.3f}  kappa={kappa:.0f}")

    as_gamma, _ = tune_gamma(
        midprices=midprices,
        dt_array=dt_array,
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
        midprices=midprices,
        dt_array=dt_array,
        gamma=as_gamma,
        sigma=sigma,
        kappa=kappa,
        A=A,
        terminal_time=terminal_time,
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

    pd.DataFrame([row]).to_csv(out_path, index=False)
    print(f"[{datetime.now()}] Saved: {out_path}")
    print(
        f"  Sharpe={row['Sharpe']:+.4f}  MaxDD={row['Max DD']:.4f}  "
        f"PnL={row['Final PnL']:+.4f}"
    )


if __name__ == "__main__":
    main()
