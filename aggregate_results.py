"""
aggregate_results.py
--------------------
Combines all per-day CSVs from results/day_*.csv into a single
summary table and plots, replicating the notebook's Section 4.

Run after all SLURM array jobs have finished:
    python aggregate_results.py
"""

import sys
import pathlib
import glob
import pandas as pd
import matplotlib.pyplot as plt

# ── Project root ──────────────────────────────────────────────────────────────
PROJECT_ROOT = next(
    (p for p in [pathlib.Path.cwd(), *pathlib.Path.cwd().parents]
     if (p / "procs").exists()),
    pathlib.Path.cwd()
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from procs.gym.experiment_config import ReplayExperimentConfig

cfg = ReplayExperimentConfig()

# ── Load all per-day CSVs ─────────────────────────────────────────────────────
results_dir = PROJECT_ROOT / "results"
csv_files = sorted(glob.glob(str(results_dir / "day_*.csv")))

if not csv_files:
    print("No day_*.csv files found in results/. Have the jobs finished?")
    sys.exit(1)

df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
df = df.set_index("Day").sort_index()
print(f"Loaded {len(df)} days.\n")

# ── Summary table ─────────────────────────────────────────────────────────────
summary = pd.DataFrame({
    col: [df[col].mean(), df[col].std(ddof=0), df[col].median()]
    for col in df.columns
}, index=["Mean", "Std", "Median"])

full = pd.concat([df, summary])
print(full.to_string(float_format="%.6f"))

# Save summary CSV
summary_path = cfg.result_path(f"as_baseline_{len(df)}day_results.csv")
df.to_csv(summary_path)
print(f"\nSaved per-day results to {summary_path}")

# ── Plots ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
metrics = ["Sharpe", "Sortino", "Max DD", "P&L-to-MAP", "Final PnL", "sigma"]
colors  = ["steelblue", "seagreen", "indianred", "mediumpurple", "darkorange", "grey"]

for ax, metric, color in zip(axes.flat, metrics, colors):
    ax.bar(range(len(df)), df[metric], color=color, alpha=0.8)
    ax.axhline(y=df[metric].mean(), color="k", ls="--", lw=1,
               label=f"mean={df[metric].mean():.4f}")
    ax.set_title(metric)
    ax.set_xlabel("Day")
    ax.set_ylabel(metric)
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df.index, rotation=45, ha="right", fontsize=7)
    ax.legend(fontsize=8)

plt.suptitle(
    f"A-S Baseline (per-day calibration) - {len(df)} Available Days {cfg.pair}",
    fontsize=14
)
plt.tight_layout()

plot_path = results_dir / f"as_baseline_{len(df)}day_plot.png"
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
print(f"Saved plot to {plot_path}")
plt.show()
