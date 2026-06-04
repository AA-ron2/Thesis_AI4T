from __future__ import annotations

import argparse
import os

import pandas as pd
from stable_baselines3 import PPO

from common import archive_once, load_context, load_train_arrays, print_context_banner, write_metadata
from procs.gym.cvar_lagrangian import CVaRLagrangianCallback, DrawdownCostWrapper
from procs.gym.notebook_support import build_formula_multi_day_replay_env, make_vecnorm
from procs.gym.sb3_wrapper import StableBaselinesTradingEnvironment
from procs.rewards import CjMmCriterion


def main() -> None:
    parser = argparse.ArgumentParser(description="Train formula-based B3 PPO with rollout-window CVaR Lagrangian.")
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--cvar-alpha", type=float, default=None)
    parser.add_argument("--lambda-lr", type=float, default=0.01)
    parser.add_argument("--lambda-max", type=float, default=500.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ctx = load_context()
    print_context_banner(ctx, "train_formula_b3")
    train_S, train_dt, train_dates, train_features = load_train_arrays(ctx)
    train_snapshots = sum(len(S) for S in train_S)
    total_timesteps = args.total_timesteps or int(os.environ.get("TOTAL_TIMESTEPS", 0)) or max(train_snapshots, 1_000_000)
    cvar_alpha = args.cvar_alpha if args.cvar_alpha is not None else float(os.environ.get("CVAR_ALPHA", 0.2))

    threshold_path = ctx.cfg.result_path("cvar_threshold_d.txt")
    if not threshold_path.exists():
        raise FileNotFoundError("Missing cvar_threshold_d.txt. Run calibrate_cvar_b3.py first.")
    d = float(threshold_path.read_text(encoding="utf-8").splitlines()[0])

    best_alpha_path = ctx.cfg.result_path("b2_best_alpha.txt")
    if not best_alpha_path.exists():
        raise FileNotFoundError("Missing b2_best_alpha.txt. Run aggregate_final_results.py --select-b2 first.")
    best_alpha = float(best_alpha_path.read_text(encoding="utf-8").strip())
    model_b2_best = PPO.load(str(ctx.cfg.model_path(f"ppo_b2_alpha{best_alpha}_doge")), device="cpu")

    env = build_formula_multi_day_replay_env(
        train_S,
        train_dt,
        ctx.cfg,
        daily_market_features=train_features,
        reward_fn=CjMmCriterion(per_step_inventory_aversion=ctx.cfg.phi),
        mode="sequential",
        **ctx.formula_kwargs,
    )
    sb3_env = StableBaselinesTradingEnvironment(env)
    cost_wrapper = DrawdownCostWrapper(sb3_env)
    vecnorm = make_vecnorm(cost_wrapper, ctx.cfg, training=True, norm_reward=False)

    callback = CVaRLagrangianCallback(
        cost_wrapper=cost_wrapper,
        cvar_alpha=cvar_alpha,
        dd_threshold=d,
        lr_lambda=args.lambda_lr,
        lambda_max=args.lambda_max,
        verbose=1,
    )
    model = PPO(
        "MlpPolicy",
        vecnorm,
        **ctx.cfg.ppo_kwargs(),
        tensorboard_log=str(ctx.cfg.repo_root / "tb_logs" / "b3"),
        verbose=1,
        device="cpu",
        seed=args.seed,
    )
    model.set_parameters(model_b2_best.get_parameters())
    print(f"B3 warm-started from B2 alpha={best_alpha}")
    model.learn(total_timesteps=total_timesteps, callback=callback)

    model_path = ctx.cfg.model_path("ppo_b3_doge").with_suffix(".zip")
    vecnorm_path = ctx.cfg.vecnorm_path("vecnorm_b3")
    archive_once(model_path, ctx.cfg.model_path("ppo_b3_doge_direct_depth").with_suffix(".zip"))
    archive_once(vecnorm_path, ctx.cfg.vecnorm_path("vecnorm_b3_direct_depth"))
    model.save(str(ctx.cfg.model_path("ppo_b3_doge")))
    vecnorm.save(str(vecnorm_path))

    history_path = ctx.cfg.result_path("b3_cvar_lagrangian_history.csv")
    pd.DataFrame(
        {
            "rollout": range(len(callback.lambda_history)),
            "lambda": callback.lambda_history,
            "cvar": callback.cvar_history,
            "threshold_d": d,
        }
    ).to_csv(history_path, index=False)

    write_metadata(
        "train_formula_b3",
        {
            "model": "b3",
            "result_type": "formula_as_parameter_ppo_rollout_window_cvar",
            "model_path": model_path,
            "vecnorm_path": vecnorm_path,
            "history_path": history_path,
            "threshold_d": d,
            "constraint_scope": "rollout_window",
            "warm_start_alpha": best_alpha,
            "total_timesteps": total_timesteps,
            "train_dates": train_dates,
            "formula_kwargs": ctx.formula_kwargs,
            "ppo_kwargs": ctx.cfg.ppo_kwargs(),
        },
    )
    print(f"Saved B3 model -> {model_path}")
    print(f"Saved B3 VecNormalize -> {vecnorm_path}")
    print(f"Saved B3 CVaR history -> {history_path}")


if __name__ == "__main__":
    main()

