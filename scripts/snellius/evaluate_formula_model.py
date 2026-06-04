from __future__ import annotations

import argparse
import os

from stable_baselines3 import PPO

from common import eval_part_path, load_context, load_test_day, model_key, print_context_banner, write_metadata
from procs.gym.notebook_support import evaluate_formula_rl_per_day


def _day_index_from_args(value: int | None) -> int:
    if value is not None:
        return value
    task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    if task_id is None:
        raise ValueError("--day-index is required when SLURM_ARRAY_TASK_ID is not set.")
    return int(task_id)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate one formula-AS model on one held-out test day.")
    parser.add_argument("--model", choices=("b1", "b2", "b3"), required=True)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--day-index", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    day_index = _day_index_from_args(args.day_index)
    if args.model == "b2" and args.alpha is None:
        raise ValueError("--alpha is required for --model b2")

    ctx = load_context()
    print_context_banner(ctx, f"evaluate_formula_{model_key(args.model, args.alpha)}_day{day_index}")
    S, dt, date, features = load_test_day(ctx, day_index)

    if args.model == "b1":
        model_path = ctx.cfg.model_path("ppo_b1_doge")
        vecnorm_path = ctx.cfg.vecnorm_path("vecnorm_b1")
    elif args.model == "b2":
        model_path = ctx.cfg.model_path(f"ppo_b2_alpha{args.alpha}_doge")
        vecnorm_path = ctx.cfg.vecnorm_path(f"vecnorm_b2_alpha{args.alpha}")
    else:
        model_path = ctx.cfg.model_path("ppo_b3_doge")
        vecnorm_path = ctx.cfg.vecnorm_path("vecnorm_b3")

    model = PPO.load(str(model_path), device="cpu")
    df = evaluate_formula_rl_per_day(
        model=model,
        vecnorm_path=vecnorm_path,
        test_S=[S],
        test_dt=[dt],
        test_dates=[date],
        test_market_features=[features],
        config=ctx.cfg,
        seed=args.seed,
        num_rollouts=ctx.cfg.evaluation_rollouts,
        **ctx.formula_kwargs,
    )
    df["Model"] = args.model.upper()
    if args.alpha is not None:
        df["Alpha"] = args.alpha
    df["Result Type"] = "formula_as_parameter_ppo"

    out_path = eval_part_path(ctx.cfg, model=args.model, alpha=args.alpha, date=date)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path)

    write_metadata(
        f"evaluate_formula_{model_key(args.model, args.alpha)}",
        {
            "model": args.model,
            "alpha": args.alpha,
            "day_index": day_index,
            "date": date,
            "output_path": out_path,
            "model_path": model_path,
            "vecnorm_path": vecnorm_path,
            "rollouts": ctx.cfg.evaluation_rollouts,
            "result_type": "formula_as_parameter_ppo",
        },
    )
    print(f"Saved evaluation part -> {out_path}")
    print(df.to_string())


if __name__ == "__main__":
    main()

