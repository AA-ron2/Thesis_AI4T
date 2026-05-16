"""
Snellius-only workflow CLI for per-day A-S baseline calibration.

Usage examples
--------------
python baseline_snellius.py prepare --project-dir /home/user/thesis --data-dir /scratch-shared/user/datasets --conda-env mysimenv
python baseline_snellius.py submit --project-dir /home/user/thesis --conda-env mysimenv --max-concurrent 6
python baseline_snellius.py status
python baseline_snellius.py aggregate
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

DEFAULT_PAIR = "DOGEUSDT"
DEFAULT_MANIFEST_REL = Path("results") / "baseline_manifest.csv"
DEFAULT_JOB_META_REL = Path("results") / "baseline_snellius_job.json"
DEFAULT_LOG_DIR_REL = Path("logs")
MANIFEST_COLUMNS = ["day_index", "date", "input_path", "result_path"]
DAY_PATTERN = re.compile(r"(?P<date>\d{4}-\d{2}-\d{2})")


@dataclass(frozen=True)
class SubmissionMetadata:
    job_id: str
    manifest_path: str
    project_dir: str
    conda_env: str
    max_concurrent: int
    log_dir: str
    submit_command: list[str]
    submitted_at: str


def repo_root() -> Path:
    return Path(__file__).resolve().parent


def ensure_linux_absolute_path(raw_value: str, label: str) -> Path:
    if "\\" in raw_value or re.match(r"^[A-Za-z]:", raw_value):
        raise ValueError(f"{label} must be a Linux path on Snellius, got: {raw_value}")
    if not raw_value.startswith("/"):
        raise ValueError(f"{label} must be an absolute Linux path, got: {raw_value}")
    return Path(raw_value)


def resolve_output_path(project_dir: Path, raw_value: str | None, default_rel: Path) -> Path:
    if raw_value is None:
        return (project_dir / default_rel).resolve()
    path = Path(raw_value)
    if path.is_absolute():
        return path.resolve()
    return (project_dir / path).resolve()


def extract_date(path: Path) -> str:
    match = DAY_PATTERN.search(path.name)
    if match is None:
        raise ValueError(f"Could not extract YYYY-MM-DD from filename: {path.name}")
    return match.group("date")


def discover_input_files(data_dir: Path, pair: str, max_days: int | None) -> list[Path]:
    files = sorted(data_dir.glob(f"binance_book_snapshot_25_*_{pair}.csv"))
    if max_days is not None:
        files = files[:max_days]
    return files


def write_manifest(
    project_dir: Path,
    data_dir: Path,
    pair: str,
    manifest_path: Path,
    max_days: int | None,
) -> pd.DataFrame:
    files = discover_input_files(data_dir, pair=pair, max_days=max_days)
    if not files:
        raise FileNotFoundError(
            f"No files matching binance_book_snapshot_25_*_{pair}.csv in {data_dir}"
        )

    results_dir = project_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for day_index, input_path in enumerate(files):
        date = extract_date(input_path)
        result_path = results_dir / f"day_{date}.csv"
        rows.append(
            {
                "day_index": day_index,
                "date": date,
                "input_path": str(input_path.resolve()),
                "result_path": str(result_path.resolve()),
            }
        )

    manifest = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_path, index=False)
    return manifest


def load_manifest(manifest_path: Path) -> pd.DataFrame:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    manifest = pd.read_csv(manifest_path)
    missing_columns = set(MANIFEST_COLUMNS).difference(manifest.columns)
    if missing_columns:
        raise ValueError(
            f"Manifest missing required columns: {sorted(missing_columns)}"
        )
    return manifest


def print_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    summary = pd.DataFrame(
        {
            col: [df[col].mean(), df[col].std(ddof=0), df[col].median()]
            for col in df.columns
        },
        index=["Mean", "Std", "Median"],
    )
    full = pd.concat([df, summary])
    print(full.to_string(float_format="%.6f"))
    return summary


def aggregate_manifest_results(
    manifest_path: Path,
    pair: str = DEFAULT_PAIR,
    allow_partial: bool = False,
) -> tuple[pd.DataFrame, Path, Path]:
    manifest = load_manifest(manifest_path)
    manifest["result_exists"] = manifest["result_path"].map(lambda value: Path(value).exists())

    completed = manifest.loc[manifest["result_exists"]].copy()
    missing = manifest.loc[~manifest["result_exists"]].copy()

    if completed.empty:
        raise FileNotFoundError("No per-day result files found for this manifest.")
    if not allow_partial and not missing.empty:
        missing_days = ", ".join(
            f"{int(row.day_index)}:{row.date}" for row in missing.itertuples(index=False)
        )
        raise RuntimeError(
            "Aggregation requires all results by default. "
            f"Missing days: {missing_days}"
        )

    df = pd.concat(
        [pd.read_csv(result_path) for result_path in completed["result_path"]],
        ignore_index=True,
    )
    df = df.set_index("Day").sort_index()
    print(f"Loaded {len(df)} day result file(s).\n")
    print_summary_table(df)

    results_dir = Path(str(completed["result_path"].iloc[0])).resolve().parent
    summary_path = results_dir / f"as_baseline_{len(df)}day_results.csv"
    plot_path = results_dir / f"as_baseline_{len(df)}day_plot.png"
    df.to_csv(summary_path)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    metrics = ["Sharpe", "Sortino", "Max DD", "P&L-to-MAP", "Final PnL", "sigma"]
    colors = ["steelblue", "seagreen", "indianred", "mediumpurple", "darkorange", "grey"]

    for ax, metric, color in zip(axes.flat, metrics, colors):
        ax.bar(range(len(df)), df[metric], color=color, alpha=0.8)
        ax.axhline(
            y=df[metric].mean(),
            color="k",
            ls="--",
            lw=1,
            label=f"mean={df[metric].mean():.4f}",
        )
        ax.set_title(metric)
        ax.set_xlabel("Day")
        ax.set_ylabel(metric)
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels(df.index, rotation=45, ha="right", fontsize=7)
        ax.legend(fontsize=8)

    plt.suptitle(
        f"A-S Baseline (per-day calibration) - {len(df)} Available Days {pair}",
        fontsize=14,
    )
    plt.tight_layout()
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved per-day results to {summary_path}")
    print(f"Saved plot to {plot_path}")
    return df, summary_path, plot_path


def read_job_metadata(job_meta_path: Path) -> SubmissionMetadata | None:
    if not job_meta_path.exists():
        return None
    with job_meta_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return SubmissionMetadata(**payload)


def format_command(command: list[str]) -> str:
    return shlex.join(command)


def prepare_command(args: argparse.Namespace) -> None:
    project_dir = ensure_linux_absolute_path(args.project_dir, "--project-dir").resolve()
    data_dir = ensure_linux_absolute_path(args.data_dir, "--data-dir").resolve()
    if not project_dir.exists():
        raise FileNotFoundError(f"Project directory not found: {project_dir}")
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    manifest_path = resolve_output_path(project_dir, args.manifest, DEFAULT_MANIFEST_REL)
    log_dir = resolve_output_path(project_dir, None, DEFAULT_LOG_DIR_REL)
    log_dir.mkdir(parents=True, exist_ok=True)

    manifest = write_manifest(
        project_dir=project_dir,
        data_dir=data_dir,
        pair=args.pair,
        manifest_path=manifest_path,
        max_days=args.max_days,
    )

    print(f"Discovered {len(manifest)} day(s) in {data_dir}")
    print(f"Manifest written to {manifest_path}")
    print(f"Conda environment: {args.conda_env}")


def submit_command(args: argparse.Namespace) -> None:
    project_dir = ensure_linux_absolute_path(args.project_dir, "--project-dir").resolve()
    manifest_path = resolve_output_path(project_dir, args.manifest, DEFAULT_MANIFEST_REL)
    job_meta_path = resolve_output_path(project_dir, args.job_meta, DEFAULT_JOB_META_REL)
    log_dir = resolve_output_path(project_dir, args.log_dir, DEFAULT_LOG_DIR_REL)

    manifest = load_manifest(manifest_path)
    if manifest.empty:
        raise ValueError(f"Manifest is empty: {manifest_path}")

    script_path = (project_dir / "baseline_sweep.sh").resolve()
    if not script_path.exists():
        raise FileNotFoundError(f"SLURM script not found: {script_path}")

    log_dir.mkdir(parents=True, exist_ok=True)
    array_spec = f"0-{len(manifest) - 1}%{args.max_concurrent}"
    export_vars = {
        "PROJECT_DIR": str(project_dir),
        "MANIFEST_PATH": str(manifest_path),
        "CONDA_ENV": args.conda_env,
    }
    export_arg = ",".join(f"{key}={value}" for key, value in export_vars.items())
    command = [
        "sbatch",
        f"--array={array_spec}",
        f"--chdir={project_dir}",
        f"--output={log_dir / 'baseline_%A_%a.out'}",
        f"--error={log_dir / 'baseline_%A_%a.err'}",
        f"--export={export_arg}",
        str(script_path),
    ]

    print(format_command(command))
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            "sbatch submission failed.\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    output = result.stdout.strip()
    match = re.search(r"Submitted batch job (?P<job_id>\d+)", output)
    if match is None:
        raise RuntimeError(f"Could not parse job id from sbatch output: {output}")

    metadata = SubmissionMetadata(
        job_id=match.group("job_id"),
        manifest_path=str(manifest_path),
        project_dir=str(project_dir),
        conda_env=args.conda_env,
        max_concurrent=args.max_concurrent,
        log_dir=str(log_dir),
        submit_command=command,
        submitted_at=datetime.now(timezone.utc).isoformat(),
    )
    job_meta_path.parent.mkdir(parents=True, exist_ok=True)
    job_meta_path.write_text(json.dumps(asdict(metadata), indent=2), encoding="utf-8")

    print(output)
    print(f"Saved job metadata to {job_meta_path}")


def status_command(args: argparse.Namespace) -> None:
    if args.project_dir is None:
        project_dir = repo_root()
    else:
        project_dir = ensure_linux_absolute_path(args.project_dir, "--project-dir").resolve()
    manifest_path = resolve_output_path(project_dir, args.manifest, DEFAULT_MANIFEST_REL)
    job_meta_path = resolve_output_path(project_dir, args.job_meta, DEFAULT_JOB_META_REL)

    manifest = load_manifest(manifest_path)
    manifest["result_exists"] = manifest["result_path"].map(lambda value: Path(value).exists())
    completed = manifest.loc[manifest["result_exists"]]
    missing = manifest.loc[~manifest["result_exists"]]

    print(f"Manifest: {manifest_path}")
    print(f"Completed days: {len(completed)}/{len(manifest)}")
    if missing.empty:
        print("All manifest outputs are present.")
    else:
        print("Missing day outputs:")
        for row in missing.itertuples(index=False):
            print(f"  [{int(row.day_index)}] {row.date}")

    metadata = read_job_metadata(job_meta_path)
    if metadata is None:
        print(f"No job metadata found at {job_meta_path}")
        return

    command = ["squeue", "-j", metadata.job_id, "--noheader", "-o", "%T"]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        print(f"Could not query squeue for job {metadata.job_id}: {result.stderr.strip()}")
        return

    states = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if states:
        unique_states = ", ".join(sorted(set(states)))
        print(f"SLURM job {metadata.job_id} is still active: {unique_states}")
    else:
        print(f"SLURM job {metadata.job_id} is not active in squeue.")


def aggregate_command(args: argparse.Namespace) -> None:
    if args.project_dir is None:
        project_dir = repo_root()
    else:
        project_dir = ensure_linux_absolute_path(args.project_dir, "--project-dir").resolve()
    manifest_path = resolve_output_path(project_dir, args.manifest, DEFAULT_MANIFEST_REL)
    aggregate_manifest_results(
        manifest_path=manifest_path,
        pair=args.pair,
        allow_partial=args.allow_partial,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--project-dir", required=True)
    prepare.add_argument("--data-dir", required=True)
    prepare.add_argument("--conda-env", required=True)
    prepare.add_argument("--pair", default=DEFAULT_PAIR)
    prepare.add_argument("--max-days", type=int, default=None)
    prepare.add_argument("--manifest", default=None)
    prepare.set_defaults(func=prepare_command)

    submit = subparsers.add_parser("submit")
    submit.add_argument("--project-dir", required=True)
    submit.add_argument("--conda-env", required=True)
    submit.add_argument("--manifest", default=None)
    submit.add_argument("--job-meta", default=None)
    submit.add_argument("--log-dir", default=None)
    submit.add_argument("--max-concurrent", type=int, default=6)
    submit.set_defaults(func=submit_command)

    status = subparsers.add_parser("status")
    status.add_argument("--project-dir", default=None)
    status.add_argument("--manifest", default=None)
    status.add_argument("--job-meta", default=None)
    status.set_defaults(func=status_command)

    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--project-dir", default=None)
    aggregate.add_argument("--manifest", default=None)
    aggregate.add_argument("--pair", default=DEFAULT_PAIR)
    aggregate.add_argument("--allow-partial", action="store_true")
    aggregate.set_defaults(func=aggregate_command)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
