from __future__ import annotations

import argparse
import os

from stable_baselines3 import PPO

from common import archive_once, load_context, load_train_arrays, print_context_banner, write_metadata
from procs.gym.notebook_support import build_formula_multi_day_replay_env, make_vecnorm
from procs.gym.sb3_wrapper import StableBaselinesTradingEnvironment
from procs.rewards import CjMmCriterion


def main() -> None:
    parser = argparse.ArgumentParser(description="Train formula-based B1 PPO.")
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ctx = load_context()
    print_context_banner(ctx, "train_formula_b1")
    train_S, train_dt, train_dates, train_features = load_train_arrays(ctx)
    train_snapshots = sum(len(S) for S in train_S)
    total_timesteps = args.total_timesteps or int(os.environ.get("TOTAL_TIMESTEPS", 0)) or max(train_snapshots, 1_000_000)

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
    vecnorm = make_vecnorm(sb3_env, ctx.cfg, training=True, norm_reward=True)

    model = PPO(
        "MlpPolicy",
        vecnorm,
        **ctx.cfg.ppo_kwargs(),
        tensorboard_log=str(ctx.cfg.repo_root / "tb_logs" / "b1"),
        verbose=1,
        device="cpu",
        seed=args.seed,
    )
    model.learn(total_timesteps=total_timesteps)

    model_path = ctx.cfg.model_path("ppo_b1_doge").with_suffix(".zip")
    vecnorm_path = ctx.cfg.vecnorm_path("vecnorm_b1")
    archive_once(model_path, ctx.cfg.model_path("ppo_b1_doge_direct_depth").with_suffix(".zip"))
    archive_once(vecnorm_path, ctx.cfg.vecnorm_path("vecnorm_b1_direct_depth"))
    model.save(str(ctx.cfg.model_path("ppo_b1_doge")))
    vecnorm.save(str(vecnorm_path))

    write_metadata(
        "train_formula_b1",
        {
            "model": "b1",
            "result_type": "formula_as_parameter_ppo",
            "model_path": model_path,
            "vecnorm_path": vecnorm_path,
            "total_timesteps": total_timesteps,
            "train_dates": train_dates,
            "formula_kwargs": ctx.formula_kwargs,
            "ppo_kwargs": ctx.cfg.ppo_kwargs(),
        },
    )
    print(f"Saved B1 model -> {model_path}")
    print(f"Saved B1 VecNormalize -> {vecnorm_path}")


if __name__ == "__main__":
    main()

