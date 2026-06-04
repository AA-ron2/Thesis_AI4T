from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from common import load_context, print_context_banner, write_metadata
from procs.gym.formula_as import FORMULA_FEATURE_NAMES
from procs.gym.notebook_support import build_formula_replay_env, freeze_vecnorm
from procs.gym.sb3_wrapper import StableBaselinesTradingEnvironment
from procs.rewards import CjMmCriterion


def main() -> None:
    parser = argparse.ArgumentParser(description="Optional SHAP diagnostics for B1 formula policy.")
    parser.add_argument("--samples", type=int, default=150)
    parser.add_argument("--background", type=int, default=30)
    parser.add_argument("--nsamples", type=int, default=100)
    args = parser.parse_args()

    try:
        import shap
    except ImportError:
        ctx = load_context()
        write_metadata("shap_diagnostics_skipped", {"reason": "shap_not_installed"})
        print("SHAP is not installed; skipping diagnostics without failing the pipeline.")
        return

    ctx = load_context()
    print_context_banner(ctx, "run_shap_diagnostics")
    row = ctx.train_manifest.iloc[0]
    from procs.gym.data_loader import load_single_day_with_features

    S, dt, _, features = load_single_day_with_features(str(row["input_path"]), tick_size=ctx.cfg.tick_size)
    env = build_formula_replay_env(
        S,
        dt,
        ctx.cfg,
        sigma=ctx.formula_kwargs["sigma"],
        market_features=features,
        reward_fn=CjMmCriterion(per_step_inventory_aversion=ctx.cfg.phi),
        **{k: v for k, v in ctx.formula_kwargs.items() if k != "sigma"},
    )
    sb3_env = StableBaselinesTradingEnvironment(env)
    eval_vn = freeze_vecnorm(ctx.cfg.vecnorm_path("vecnorm_b1"), sb3_env, ctx.cfg, norm_reward=False)
    model = PPO.load(str(ctx.cfg.model_path("ppo_b1_doge")), device="cpu")

    obs, _ = env.reset(seed=ctx.cfg.evaluation_seed)
    observations = []
    for _ in range(args.samples):
        observations.append(obs[0].copy())
        action, _ = model.predict(eval_vn.normalize_obs(obs)[0], deterministic=True)
        obs, _, terminated, _, _ = env.step(action.reshape(1, -1))
        if bool(terminated[0]):
            break

    X = np.asarray(observations, dtype=np.float32)
    if len(X) <= args.background:
        raise RuntimeError("Not enough observations collected for SHAP diagnostics.")

    background = X[: args.background]
    explain = X[args.background :]

    def predict_fn(raw_x):
        raw_x = np.asarray(raw_x, dtype=np.float32)
        norm_x = eval_vn.normalize_obs(raw_x)
        actions = [model.predict(row, deterministic=True)[0] for row in norm_x]
        return np.asarray(actions)

    explainer = shap.KernelExplainer(predict_fn, background)
    shap_values = explainer.shap_values(explain, nsamples=args.nsamples)
    if isinstance(shap_values, list):
        values = np.stack(shap_values, axis=0)
    else:
        values = np.asarray(shap_values)
        if values.ndim == 3:
            values = np.moveaxis(values, -1, 0)

    rows = []
    output_names = ["u_gamma", "u_skew"]
    for output_idx, output_name in enumerate(output_names[: values.shape[0]]):
        mean_abs = np.abs(values[output_idx]).mean(axis=0)
        for feature, score in zip(FORMULA_FEATURE_NAMES, mean_abs):
            rows.append({"output": output_name, "feature": feature, "mean_abs_shap": float(score)})

    out_path = ctx.cfg.result_path("b1_shap_feature_importance.csv")
    pd.DataFrame(rows).to_csv(out_path, index=False)
    write_metadata(
        "shap_diagnostics",
        {
            "output_path": out_path,
            "samples": len(X),
            "background": args.background,
            "nsamples": args.nsamples,
        },
    )
    print(f"Saved SHAP feature importance -> {out_path}")


if __name__ == "__main__":
    main()

