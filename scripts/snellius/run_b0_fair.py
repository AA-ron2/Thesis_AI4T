from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from common import load_context, print_context_banner, write_metadata
from procs.gym.calibration import calibrate_from_arrays, tune_gamma
from procs.gym.data_loader import load_single_day
from procs.gym.helpers.fast_rollout import fast_simulate


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fair train-only B0 analytical A-S baseline.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ctx = load_context()
    print_context_banner(ctx, "run_b0_fair")

    gamma_candidates = []
    train_rows = []
    for row in ctx.train_manifest.itertuples(index=False):
        S, dt, _ = load_single_day(row.input_path)
        sigma, A, kappa = calibrate_from_arrays(S, dt, tick_size=ctx.cfg.tick_size)
        best_gamma, _ = tune_gamma(
            S,
            dt,
            sigma=sigma,
            kappa=kappa,
            A=A,
            tick_size=ctx.cfg.tick_size,
            Q_MAX=ctx.cfg.q_max,
            gamma_range=ctx.cfg.as_gamma_range,
            n_trials=ctx.cfg.as_gamma_trials,
            num_trajectories=ctx.cfg.as_gamma_num_trajectories,
            seed=args.seed,
            verbose=False,
        )
        gamma_candidates.append(float(best_gamma))
        train_rows.append(
            {
                "Day": row.date,
                "sigma": float(sigma),
                "A": float(A),
                "kappa": float(kappa),
                "gamma_candidate": float(best_gamma),
            }
        )
        print(f"{row.date}: gamma={best_gamma:.6f}, sigma={sigma:.8f}, A={A:.6f}, kappa={kappa:.2f}")

    df_train = pd.DataFrame(train_rows)
    gamma_fixed = float(np.mean(gamma_candidates))
    sigma_train = float(df_train["sigma"].median())
    A_train = float(df_train["A"].median())
    kappa_train = float(df_train["kappa"].median())

    df_train.to_csv(ctx.cfg.result_path("b0_train_calibration_params.csv"), index=False)
    ctx.cfg.result_path("b0_gamma_fixed.txt").write_text(str(gamma_fixed), encoding="utf-8")

    rows = []
    for row in ctx.test_manifest.itertuples(index=False):
        S, dt, _ = load_single_day(row.input_path)
        stats = fast_simulate(
            midprices=S,
            dt_array=dt,
            gamma=gamma_fixed,
            sigma=sigma_train,
            kappa=kappa_train,
            A=A_train,
            terminal_time=float(dt.sum()),
            tick_size=ctx.cfg.tick_size,
            Q_MAX=ctx.cfg.q_max,
            num_trajectories=ctx.cfg.evaluation_rollouts,
            seed=args.seed,
            use_linear_approximation=False,
        )
        rows.append(
            {
                "Day": str(row.date),
                "Sharpe": float(stats["sharpe"].mean()),
                "Sortino": float(stats["sortino"].mean()),
                "Max DD": float(stats["max_drawdown"].mean()),
                "P&L-to-MAP": float(stats["pnl_to_map"].mean()),
                "Final PnL": float(stats["total_pnl"].mean()),
                "Mean |q|": float(stats["mean_abs_q"].mean()),
                "Near Cap Fraction": float(stats["near_cap_fraction"].mean()),
                "Rollouts": float(ctx.cfg.evaluation_rollouts),
                "Result Type": "train_only_fair_b0",
            }
        )
        print(f"{row.date}: Sharpe={rows[-1]['Sharpe']:.4f}, MaxDD={rows[-1]['Max DD']:.6f}")

    df = pd.DataFrame(rows)
    out_path = ctx.cfg.result_path("b0_test_results.csv")
    df.to_csv(out_path, index=False)
    write_metadata(
        "run_b0_fair",
        {
            "output_path": out_path,
            "gamma_fixed": gamma_fixed,
            "sigma_train": sigma_train,
            "A_train": A_train,
            "kappa_train": kappa_train,
            "result_type": "train_only_fair_b0",
        },
    )
    print(f"Saved B0 results -> {out_path}")


if __name__ == "__main__":
    main()

