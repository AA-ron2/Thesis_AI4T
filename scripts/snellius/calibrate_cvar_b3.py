from __future__ import annotations

import argparse
import os

from stable_baselines3 import PPO

from common import load_context, load_train_arrays, print_context_banner, write_metadata
from procs.gym.notebook_support import calibrate_cvar_threshold_sampled_windows


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate B3 rollout-window CVaR threshold from B1 rollouts.")
    parser.add_argument("--n-windows", type=int, default=None)
    parser.add_argument("--cvar-alpha", type=float, default=None)
    parser.add_argument("--tighten", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ctx = load_context()
    print_context_banner(ctx, "calibrate_cvar_b3")
    train_S, train_dt, train_dates, train_features = load_train_arrays(ctx)
    model_b1 = PPO.load(str(ctx.cfg.model_path("ppo_b1_doge")), device="cpu")

    cvar_alpha = args.cvar_alpha if args.cvar_alpha is not None else float(os.environ.get("CVAR_ALPHA", 0.2))
    n_windows = args.n_windows if args.n_windows is not None else int(os.environ.get("CVAR_N_WINDOWS", 50))
    tighten = args.tighten if args.tighten is not None else float(os.environ.get("CVAR_TIGHTEN", 0.2))

    d, cvar_raw = calibrate_cvar_threshold_sampled_windows(
        daily_midprices=train_S,
        daily_dt_arrays=train_dt,
        daily_market_features=train_features,
        model=model_b1,
        vecnorm_path=ctx.cfg.vecnorm_path("vecnorm_b1"),
        config=ctx.cfg,
        n_steps=ctx.cfg.ppo_n_steps,
        cvar_alpha=cvar_alpha,
        n_windows=n_windows,
        tighten=tighten,
        seed=args.seed,
        verbose=True,
        sigma=ctx.formula_kwargs["sigma"],
        formula_gamma_min=ctx.formula_kwargs["gamma_min"],
        formula_gamma_max=ctx.formula_kwargs["gamma_max"],
        formula_skew_ticks_max=ctx.formula_kwargs["skew_ticks_max"],
    )

    txt_path = ctx.cfg.result_path("cvar_threshold_d.txt")
    json_path = ctx.cfg.result_path("cvar_threshold_b3.json")
    txt_path.write_text(f"{d}\n{cvar_raw}\n", encoding="utf-8")
    json_path.write_text(
        (
            "{\n"
            f'  "constraint_scope": "rollout_window",\n'
            f'  "n_steps": {ctx.cfg.ppo_n_steps},\n'
            f'  "n_windows": {n_windows},\n'
            f'  "cvar_alpha": {cvar_alpha},\n'
            f'  "tighten": {tighten},\n'
            f'  "threshold_d": {d},\n'
            f'  "raw_cvar": {cvar_raw}\n'
            "}\n"
        ),
        encoding="utf-8",
    )
    write_metadata(
        "calibrate_cvar_b3",
        {
            "constraint_scope": "rollout_window",
            "threshold_path": txt_path,
            "threshold_json": json_path,
            "threshold_d": d,
            "raw_cvar": cvar_raw,
            "cvar_alpha": cvar_alpha,
            "n_windows": n_windows,
            "n_steps": ctx.cfg.ppo_n_steps,
            "train_dates": train_dates,
        },
    )
    print(f"Saved B3 CVaR threshold -> {txt_path}")
    print(f"Saved B3 CVaR threshold metadata -> {json_path}")


if __name__ == "__main__":
    main()

