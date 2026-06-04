from __future__ import annotations

import argparse
import os

from stable_baselines3 import PPO

from common import load_context, print_context_banner, write_metadata
from procs.gym.formula_as import FORMULA_FEATURE_NAMES
from procs.gym.notebook_support import (
    build_formula_multi_day_replay_env,
    evaluate_formula_rl_per_day,
    make_vecnorm,
)
from procs.gym.sb3_wrapper import StableBaselinesTradingEnvironment
from procs.rewards import CjMmCriterion

FEATURE_SETS = {
    "base_state": ("inventory_norm", "time_remaining_frac", "log_mid_rel"),
    "base_plus_vol": ("inventory_norm", "time_remaining_frac", "log_mid_rel", "rolling_vol_ticks"),
    "full": tuple(FORMULA_FEATURE_NAMES),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one formula-AS feature ablation job.")
    parser.add_argument("--feature-set", choices=tuple(FEATURE_SETS), required=True)
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ctx = load_context()
    print_context_banner(ctx, f"run_feature_ablation_{args.feature_set}")
    features = FEATURE_SETS[args.feature_set]
    total_timesteps = args.total_timesteps or int(os.environ.get("ABLATION_TIMESTEPS", 100_000))

    train_rows = ctx.train_manifest.iloc[:4]
    val_rows = ctx.train_manifest.iloc[4:6]
    train_S, train_dt, train_features = [], [], []
    val_S, val_dt, val_dates, val_features = [], [], [], []
    from procs.gym.data_loader import load_single_day_with_features

    for row in train_rows.itertuples(index=False):
        S, dt, _, feats = load_single_day_with_features(row.input_path, tick_size=ctx.cfg.tick_size)
        train_S.append(S)
        train_dt.append(dt)
        train_features.append(feats)
    for row in val_rows.itertuples(index=False):
        S, dt, _, feats = load_single_day_with_features(row.input_path, tick_size=ctx.cfg.tick_size)
        val_S.append(S)
        val_dt.append(dt)
        val_dates.append(str(row.date))
        val_features.append(feats)

    env = build_formula_multi_day_replay_env(
        train_S,
        train_dt,
        ctx.cfg,
        daily_market_features=train_features,
        reward_fn=CjMmCriterion(per_step_inventory_aversion=ctx.cfg.phi),
        mode="sequential",
        enabled_features=features,
        **ctx.formula_kwargs,
    )
    sb3_env = StableBaselinesTradingEnvironment(env)
    vecnorm = make_vecnorm(sb3_env, ctx.cfg, training=True, norm_reward=True)
    model = PPO(
        "MlpPolicy",
        vecnorm,
        **ctx.cfg.ppo_kwargs(),
        tensorboard_log=str(ctx.cfg.repo_root / "tb_logs" / f"ablation_{args.feature_set}"),
        verbose=1,
        device="cpu",
        seed=args.seed,
    )
    model.learn(total_timesteps=total_timesteps)

    model_stem = f"ppo_ablation_{args.feature_set}"
    vecnorm_stem = f"vecnorm_ablation_{args.feature_set}"
    model.save(str(ctx.cfg.model_path(model_stem)))
    vecnorm.save(str(ctx.cfg.vecnorm_path(vecnorm_stem)))

    df = evaluate_formula_rl_per_day(
        model=model,
        vecnorm_path=ctx.cfg.vecnorm_path(vecnorm_stem),
        test_S=val_S,
        test_dt=val_dt,
        test_dates=val_dates,
        test_market_features=val_features,
        config=ctx.cfg,
        seed=args.seed,
        num_rollouts=ctx.cfg.evaluation_rollouts,
        enabled_features=features,
        **ctx.formula_kwargs,
    )
    out_dir = ctx.cfg.results_dir / "ablation"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.feature_set}_validation_results.csv"
    df.to_csv(out_path)
    write_metadata(
        f"feature_ablation_{args.feature_set}",
        {
            "feature_set": args.feature_set,
            "features": features,
            "total_timesteps": total_timesteps,
            "output_path": out_path,
            "model_path": ctx.cfg.model_path(model_stem).with_suffix(".zip"),
            "vecnorm_path": ctx.cfg.vecnorm_path(vecnorm_stem),
        },
    )
    print(f"Saved ablation validation -> {out_path}")


if __name__ == "__main__":
    main()

