from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd

from common import (
    b2_alpha_result_path,
    eval_part_path,
    load_context,
    parse_alpha_list,
    write_metadata,
)

SUMMARY_METRICS = ("Sharpe", "Sortino", "Max DD", "P&L-to-MAP", "Final PnL", "Mean |q|")
DAYS_BEST_METRICS = {
    "Sharpe": True,
    "Sortino": True,
    "Max DD": False,
    "P&L-to-MAP": True,
}


def _read_part(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "Day" not in frame.columns:
        raise ValueError(f"Evaluation part missing Day column: {path}")
    return frame


def aggregate_model_parts(
    *,
    model: str,
    alpha: float | None = None,
    allow_partial: bool = False,
) -> pd.DataFrame:
    ctx = load_context()
    frames = []
    missing = []
    for row in ctx.test_manifest.itertuples(index=False):
        path = eval_part_path(ctx.cfg, model=model, alpha=alpha, date=str(row.date))
        if path.exists():
            frames.append(_read_part(path))
        else:
            missing.append((row.split_index, row.date, path))

    if missing and not allow_partial:
        missing_text = ", ".join(f"{idx}:{date}" for idx, date, _ in missing)
        raise FileNotFoundError(f"Missing evaluation parts for {model}: {missing_text}")
    if not frames:
        raise FileNotFoundError(f"No evaluation parts found for model={model}, alpha={alpha}")

    df = pd.concat(frames, ignore_index=True).sort_values("Day").reset_index(drop=True)
    if model == "b1":
        out_path = ctx.cfg.result_path("b1_test_results.csv")
    elif model == "b2":
        if alpha is None:
            raise ValueError("alpha is required to aggregate B2 alpha results")
        out_path = b2_alpha_result_path(ctx.cfg, alpha)
    elif model == "b3":
        out_path = ctx.cfg.result_path("b3_test_results.csv")
    else:
        raise ValueError(f"Unknown model: {model}")
    df.to_csv(out_path, index=False)
    write_metadata(
        f"aggregate_{model}" if alpha is None else f"aggregate_{model}_alpha{alpha}",
        {
            "model": model,
            "alpha": alpha,
            "output_path": out_path,
            "rows": len(df),
            "allow_partial": allow_partial,
            "missing": [{"day_index": idx, "date": date, "path": path} for idx, date, path in missing],
            "result_type": "formula_as_parameter_ppo",
        },
    )
    print(f"Saved aggregated {model} results -> {out_path}")
    return df


def select_b2(alphas: list[float]) -> float:
    ctx = load_context()
    rows = []
    for alpha in alphas:
        path = b2_alpha_result_path(ctx.cfg, alpha)
        if not path.exists():
            aggregate_model_parts(model="b2", alpha=alpha)
        df = pd.read_csv(path)
        rows.append(
            {
                "alpha": alpha,
                "mean_sharpe": float(df["Sharpe"].mean()),
                "mean_maxdd": float(df["Max DD"].mean()),
                "mean_pnl": float(df["Final PnL"].mean()),
            }
        )

    summary = pd.DataFrame(rows).set_index("alpha")
    positive = summary[summary["mean_sharpe"] > 0]
    if len(positive):
        best_alpha = float(positive.sort_values(["mean_maxdd", "mean_sharpe"], ascending=[True, False]).index[0])
    else:
        best_alpha = float(summary.sort_index().index[0])

    selected_path = b2_alpha_result_path(ctx.cfg, best_alpha)
    shutil.copy2(selected_path, ctx.cfg.result_path("b2_test_results.csv"))
    ctx.cfg.result_path("b2_best_alpha.txt").write_text(str(best_alpha), encoding="utf-8")
    summary.to_csv(ctx.cfg.result_path("b2_alpha_sweep_summary.csv"))
    write_metadata(
        "select_b2",
        {
            "best_alpha": best_alpha,
            "alpha_summary": summary.reset_index().to_dict(orient="records"),
            "selected_result": ctx.cfg.result_path("b2_test_results.csv"),
        },
    )
    print("=== B2 alpha summary ===")
    print(summary.to_string(float_format="{:.6f}".format))
    print(f"Selected B2 alpha={best_alpha}")
    return best_alpha


def write_comparison() -> None:
    ctx = load_context()
    paths = {
        "B0": ctx.cfg.result_path("b0_test_results.csv"),
        "B1": ctx.cfg.result_path("b1_test_results.csv"),
        "B2": ctx.cfg.result_path("b2_test_results.csv"),
        "B3": ctx.cfg.result_path("b3_test_results.csv"),
    }
    missing = [f"{name}:{path}" for name, path in paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing comparison inputs. " + ", ".join(missing))

    dfs = {name: pd.read_csv(path).sort_values("Day").reset_index(drop=True) for name, path in paths.items()}
    base_dates = dfs["B0"]["Day"].astype(str).tolist()
    for name, df in dfs.items():
        dates = df["Day"].astype(str).tolist()
        if dates != base_dates:
            raise ValueError(f"{name} dates do not align with B0.")

    rows = []
    for metric, higher_better in DAYS_BEST_METRICS.items():
        values = pd.DataFrame({name: df[metric] for name, df in dfs.items()})
        winners = values.idxmax(axis=1) if higher_better else values.idxmin(axis=1)
        counts = winners.value_counts().reindex(dfs.keys(), fill_value=0)
        row = {"Metric": metric}
        row.update(counts.to_dict())
        rows.append(row)
    comparison = pd.DataFrame(rows)
    comparison.to_csv(ctx.cfg.result_path("comparison_table.csv"), index=False)

    summary_rows = []
    for name, df in dfs.items():
        row = {"Model": name}
        for metric in SUMMARY_METRICS:
            if metric in df:
                row[f"{metric} Mean"] = float(df[metric].mean())
                row[f"{metric} Median"] = float(df[metric].median())
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(ctx.cfg.result_path("overall_summary_table.csv"), index=False)

    write_metadata(
        "final_comparison",
        {
            "comparison_table": ctx.cfg.result_path("comparison_table.csv"),
            "overall_summary": ctx.cfg.result_path("overall_summary_table.csv"),
            "dates": base_dates,
            "result_type": "formula_as_parameter_ppo_comparison",
        },
    )
    print(f"Saved comparison table -> {ctx.cfg.result_path('comparison_table.csv')}")
    print(f"Saved overall summary -> {ctx.cfg.result_path('overall_summary_table.csv')}")


def aggregate_ablation() -> None:
    ctx = load_context()
    rows = []
    for path in sorted((ctx.cfg.results_dir / "ablation").glob("*_validation_results.csv")):
        feature_set = path.name.replace("_validation_results.csv", "")
        df = pd.read_csv(path)
        row = {"feature_set": feature_set}
        for metric in SUMMARY_METRICS:
            if metric in df:
                row[f"mean_{metric}"] = float(df[metric].mean())
        rows.append(row)
    if not rows:
        raise FileNotFoundError("No ablation validation files found.")
    out = pd.DataFrame(rows)
    out.to_csv(ctx.cfg.result_path("formula_feature_ablation_results.csv"), index=False)
    print(f"Saved ablation summary -> {ctx.cfg.result_path('formula_feature_ablation_results.csv')}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate formula-AS Snellius outputs.")
    parser.add_argument("--model", choices=("b1", "b2", "b3"), default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--select-b2", action="store_true")
    parser.add_argument("--alphas", default=None)
    parser.add_argument("--comparison", action="store_true")
    parser.add_argument("--ablation", action="store_true")
    args = parser.parse_args()

    if args.model:
        aggregate_model_parts(model=args.model, alpha=args.alpha, allow_partial=args.allow_partial)
    if args.select_b2:
        select_b2(parse_alpha_list(args.alphas))
    if args.comparison:
        write_comparison()
    if args.ablation:
        aggregate_ablation()
    if not any([args.model, args.select_b2, args.comparison, args.ablation]):
        raise SystemExit("No action selected. Use --model, --select-b2, --comparison, or --ablation.")


if __name__ == "__main__":
    main()

