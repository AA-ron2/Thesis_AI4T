from __future__ import annotations

import json
import os
import re
import socket
import subprocess
import sys
from dataclasses import asdict, dataclass
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


def _discover_project_dir() -> Path:
    env_value = os.environ.get("PROJECT_DIR")
    if env_value:
        return Path(env_value).expanduser().resolve()
    start = Path(__file__).resolve()
    for candidate in [start, *start.parents]:
        if (candidate / "procs").exists():
            return candidate.resolve()
    return Path.cwd().resolve()


PROJECT_DIR = _discover_project_dir()
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from procs.gym.calibration import calibrate_from_arrays  # noqa: E402
from procs.gym.data_loader import (  # noqa: E402
    load_single_day_with_features,
)
from procs.gym.experiment_config import ReplayExperimentConfig  # noqa: E402
from procs.gym.formula_as import FORMULA_FEATURE_NAMES  # noqa: E402


DEFAULT_TRAIN_DAYS = 6
DEFAULT_EXPECTED_TEST_DAYS = 23
DEFAULT_ALPHAS = (0.1, 1.0, 10.0)
DEFAULT_FORMULA_GAMMA_MIN = 0.01
DEFAULT_FORMULA_GAMMA_MAX = 0.9
DEFAULT_FORMULA_SKEW_TICKS_MAX = 5.0
MANIFEST_COLUMNS = ("row_index", "split", "split_index", "date", "input_path")
DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")


@dataclass(frozen=True)
class FormulaPipelineContext:
    project_dir: Path
    data_dir: Path
    cfg: ReplayExperimentConfig
    manifest: pd.DataFrame
    train_params: dict[str, Any]
    formula_kwargs: dict[str, float]

    @property
    def train_manifest(self) -> pd.DataFrame:
        return self.manifest[self.manifest["split"] == "train"].reset_index(drop=True)

    @property
    def test_manifest(self) -> pd.DataFrame:
        return self.manifest[self.manifest["split"] == "test"].reset_index(drop=True)


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return default if value in (None, "") else int(value)


def env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return default if value in (None, "") else float(value)


def env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def parse_alpha_list(raw: str | None = None) -> list[float]:
    value = raw or os.environ.get("B2_ALPHAS")
    if not value:
        return list(DEFAULT_ALPHAS)
    return [float(part) for part in value.replace(",", " ").split()]


def require_snellius_paths() -> None:
    data_dir = os.environ.get("DATA_DIR")
    if not data_dir:
        raise RuntimeError("DATA_DIR must be exported on Snellius before running the pipeline.")
    if "\\" in data_dir or re.match(r"^[A-Za-z]:", data_dir):
        raise RuntimeError(f"DATA_DIR must be a Linux path on Snellius, got: {data_dir}")


def build_config() -> ReplayExperimentConfig:
    data_dir = os.environ.get("DATA_DIR")
    cfg = ReplayExperimentConfig(
        repo_root=PROJECT_DIR,
        datasets_subdir=data_dir or ReplayExperimentConfig().datasets_subdir,
    )
    eval_rollouts = env_int("EVALUATION_ROLLOUTS", cfg.evaluation_rollouts)
    feature_window = env_int("FEATURE_WINDOW", cfg.feature_window)
    return replace(cfg, evaluation_rollouts=eval_rollouts, feature_window=feature_window)


def manifest_path(cfg: ReplayExperimentConfig) -> Path:
    return cfg.result_path("formula_manifest.csv")


def train_params_json_path(cfg: ReplayExperimentConfig) -> Path:
    return cfg.result_path("formula_train_params.json")


def train_params_csv_path(cfg: ReplayExperimentConfig) -> Path:
    return cfg.result_path("formula_train_params.csv")


def _extract_date(path: Path) -> str:
    match = DATE_RE.search(path.name)
    if match is None:
        raise ValueError(f"Could not extract YYYY-MM-DD date from {path.name}")
    return match.group(0)


def prepare_manifest(
    cfg: ReplayExperimentConfig,
    *,
    train_days: int = DEFAULT_TRAIN_DAYS,
    expected_test_days: int = DEFAULT_EXPECTED_TEST_DAYS,
    max_days: int | None = None,
) -> pd.DataFrame:
    files = sorted(cfg.datasets_dir.glob(f"binance_book_snapshot_25_*_{cfg.pair}.csv"))
    if max_days is not None:
        files = files[:max_days]
    if not files:
        raise FileNotFoundError(f"No DOGE files found in DATA_DIR={cfg.datasets_dir}")
    if len(files) < train_days + expected_test_days:
        raise ValueError(
            f"Expected at least {train_days + expected_test_days} files "
            f"({train_days} train + {expected_test_days} test), found {len(files)}."
        )

    files = files[: train_days + expected_test_days]
    rows: list[dict[str, Any]] = []
    for row_index, path in enumerate(files):
        split = "train" if row_index < train_days else "test"
        split_index = row_index if split == "train" else row_index - train_days
        rows.append(
            {
                "row_index": row_index,
                "split": split,
                "split_index": split_index,
                "date": _extract_date(path),
                "input_path": str(path.resolve()),
            }
        )

    manifest = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    out_path = manifest_path(cfg)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(out_path, index=False)
    return manifest


def load_or_prepare_manifest(cfg: ReplayExperimentConfig) -> pd.DataFrame:
    path = manifest_path(cfg)
    if not path.exists():
        return prepare_manifest(cfg)
    manifest = pd.read_csv(path)
    missing = set(MANIFEST_COLUMNS).difference(manifest.columns)
    if missing:
        raise ValueError(f"Formula manifest missing columns: {sorted(missing)}")
    return manifest


def load_single_manifest_day(row: pd.Series, cfg: ReplayExperimentConfig):
    return load_single_day_with_features(
        str(row["input_path"]),
        tick_size=cfg.tick_size,
    )


def load_train_arrays(ctx: FormulaPipelineContext):
    train_S, train_dt, train_dates, train_features = [], [], [], []
    for row in ctx.train_manifest.itertuples(index=False):
        S, dt, _, features = load_single_day_with_features(
            row.input_path,
            tick_size=ctx.cfg.tick_size,
        )
        train_S.append(S)
        train_dt.append(dt)
        train_dates.append(str(row.date))
        train_features.append(features)
        print(f"Loaded train {row.date}: {len(S):,} snapshots")
    return train_S, train_dt, train_dates, train_features


def load_test_day(ctx: FormulaPipelineContext, day_index: int):
    test_manifest = ctx.test_manifest
    if day_index < 0 or day_index >= len(test_manifest):
        raise IndexError(f"Test day index {day_index} outside 0..{len(test_manifest)-1}")
    row = test_manifest.iloc[day_index]
    S, dt, _, features = load_single_day_with_features(
        str(row["input_path"]),
        tick_size=ctx.cfg.tick_size,
    )
    return S, dt, str(row["date"]), features


def load_or_create_train_params(
    cfg: ReplayExperimentConfig,
    manifest: pd.DataFrame,
    *,
    force: bool = False,
) -> dict[str, Any]:
    json_path = train_params_json_path(cfg)
    if json_path.exists() and not force:
        return json.loads(json_path.read_text(encoding="utf-8"))

    rows: list[dict[str, Any]] = []
    for row in manifest[manifest["split"] == "train"].itertuples(index=False):
        S, dt, _, _ = load_single_day_with_features(row.input_path, tick_size=cfg.tick_size)
        sigma, A, kappa = calibrate_from_arrays(S, dt, tick_size=cfg.tick_size)
        rows.append(
            {
                "date": str(row.date),
                "sigma": float(sigma),
                "A": float(A),
                "kappa": float(kappa),
            }
        )
        print(f"Calibrated train {row.date}: sigma={sigma:.8f}, A={A:.6f}, kappa={kappa:.2f}")

    frame = pd.DataFrame(rows)
    payload = {
        "sigma_train": float(frame["sigma"].median()),
        "A_train": float(frame["A"].median()),
        "kappa_train": float(frame["kappa"].median()),
        "aggregation": "median_over_train_days",
        "per_day": rows,
    }
    train_params_csv_path(cfg).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(train_params_csv_path(cfg), index=False)
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def formula_kwargs_from_env(train_params: dict[str, Any]) -> dict[str, float]:
    return {
        "sigma": float(train_params["sigma_train"]),
        "gamma_min": env_float("FORMULA_GAMMA_MIN", DEFAULT_FORMULA_GAMMA_MIN),
        "gamma_max": env_float("FORMULA_GAMMA_MAX", DEFAULT_FORMULA_GAMMA_MAX),
        "skew_ticks_max": env_float("FORMULA_SKEW_TICKS_MAX", DEFAULT_FORMULA_SKEW_TICKS_MAX),
    }


def load_context(*, force_prepare: bool = False) -> FormulaPipelineContext:
    require_snellius_paths()
    cfg = build_config()
    cfg.ensure_artifact_dirs()
    if force_prepare:
        manifest = prepare_manifest(cfg)
    else:
        manifest = load_or_prepare_manifest(cfg)
    train_params = load_or_create_train_params(cfg, manifest, force=force_prepare)
    cfg = replace(cfg, A=float(train_params["A_train"]), kappa=float(train_params["kappa_train"]))
    return FormulaPipelineContext(
        project_dir=PROJECT_DIR,
        data_dir=cfg.datasets_dir,
        cfg=cfg,
        manifest=manifest,
        train_params=train_params,
        formula_kwargs=formula_kwargs_from_env(train_params),
    )


def model_key(model: str, alpha: float | None = None) -> str:
    if model == "b2":
        if alpha is None:
            raise ValueError("alpha is required for B2 model keys")
        return f"b2_alpha{alpha}"
    return model


def eval_part_dir(cfg: ReplayExperimentConfig, model: str, alpha: float | None = None) -> Path:
    return cfg.results_dir / "eval_parts" / model_key(model, alpha)


def eval_part_path(
    cfg: ReplayExperimentConfig,
    *,
    model: str,
    date: str,
    alpha: float | None = None,
) -> Path:
    return eval_part_dir(cfg, model, alpha) / f"day_{date}.csv"


def b2_alpha_result_path(cfg: ReplayExperimentConfig, alpha: float) -> Path:
    return cfg.result_path(f"b2_alpha{alpha}_test_results.csv")


def get_git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_DIR,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def write_metadata(job_name: str, payload: dict[str, Any]) -> Path:
    cfg = build_config()
    meta_dir = cfg.results_dir / "job_metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = meta_dir / f"{job_name}_{job_id}_{timestamp}.json"
    base = {
        "job_name": job_name,
        "slurm_job_id": job_id,
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "timestamp_utc": timestamp,
        "hostname": socket.gethostname(),
        "project_dir": str(PROJECT_DIR),
        "data_dir": os.environ.get("DATA_DIR"),
        "conda_env": os.environ.get("CONDA_ENV"),
        "git_commit": get_git_commit(),
    }
    base.update(payload)
    path.write_text(json.dumps(_json_safe(base), indent=2), encoding="utf-8")
    return path


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "__dataclass_fields__"):
        return _json_safe(asdict(value))
    return value


def archive_once(path: Path, archive_path: Path) -> None:
    if path.exists() and not archive_path.exists():
        import shutil

        archive_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, archive_path)


def print_context_banner(ctx: FormulaPipelineContext, job_name: str) -> None:
    print(f"=== {job_name} ===")
    print(f"Project dir : {ctx.project_dir}")
    print(f"Data dir    : {ctx.data_dir}")
    print(f"Results dir : {ctx.cfg.results_dir}")
    print(f"Train days  : {len(ctx.train_manifest)}")
    print(f"Test days   : {len(ctx.test_manifest)}")
    print(f"Formula     : {ctx.formula_kwargs}")
    print(f"Features    : {list(FORMULA_FEATURE_NAMES)}")

