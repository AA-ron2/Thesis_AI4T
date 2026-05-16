from __future__ import annotations

from pathlib import Path
import shutil
import uuid

import pandas as pd
import pytest

from baseline_snellius import aggregate_manifest_results, write_manifest


def make_scratch_dir() -> Path:
    scratch = Path.cwd() / f".baseline_snellius_test_{uuid.uuid4().hex}"
    scratch.mkdir(parents=True, exist_ok=False)
    return scratch


def test_write_manifest_sorts_and_maps_days() -> None:
    scratch = make_scratch_dir()
    try:
        project_dir = scratch / "project"
        data_dir = scratch / "datasets"
        manifest_path = project_dir / "results" / "baseline_manifest.csv"

        project_dir.mkdir()
        data_dir.mkdir()
        (data_dir / "binance_book_snapshot_25_2025-01-03_DOGEUSDT.csv").write_text("", encoding="utf-8")
        (data_dir / "binance_book_snapshot_25_2025-01-01_DOGEUSDT.csv").write_text("", encoding="utf-8")
        (data_dir / "binance_book_snapshot_25_2025-01-02_DOGEUSDT.csv").write_text("", encoding="utf-8")

        manifest = write_manifest(
            project_dir=project_dir,
            data_dir=data_dir,
            pair="DOGEUSDT",
            manifest_path=manifest_path,
            max_days=None,
        )

        assert list(manifest["day_index"]) == [0, 1, 2]
        assert list(manifest["date"]) == ["2025-01-01", "2025-01-02", "2025-01-03"]
        assert manifest_path.exists()
        assert Path(manifest.loc[0, "result_path"]).name == "day_2025-01-01.csv"
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def test_aggregate_manifest_results_requires_complete_outputs() -> None:
    scratch = make_scratch_dir()
    try:
        results_dir = scratch / "results"
        results_dir.mkdir()
        manifest_path = results_dir / "baseline_manifest.csv"

        existing_result = results_dir / "day_2025-01-01.csv"
        pd.DataFrame(
            [
                {
                    "Day": "2025-01-01",
                    "Sharpe": 1.0,
                    "Sortino": 1.0,
                    "Max DD": 0.1,
                    "P&L-to-MAP": 1.5,
                    "Final PnL": 2.0,
                    "Mean |q|": 0.3,
                    "Near Cap Fraction": 0.0,
                    "sigma": 0.01,
                    "A": 0.8,
                    "kappa": 35000.0,
                    "as_gamma": 0.1,
                }
            ]
        ).to_csv(existing_result, index=False)

        pd.DataFrame(
            [
                {
                    "day_index": 0,
                    "date": "2025-01-01",
                    "input_path": "/scratch/input_1.csv",
                    "result_path": str(existing_result),
                },
                {
                    "day_index": 1,
                    "date": "2025-01-02",
                    "input_path": "/scratch/input_2.csv",
                    "result_path": str(results_dir / "day_2025-01-02.csv"),
                },
            ]
        ).to_csv(manifest_path, index=False)

        with pytest.raises(RuntimeError, match="Missing days"):
            aggregate_manifest_results(manifest_path)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def test_aggregate_manifest_results_writes_summary_and_plot() -> None:
    scratch = make_scratch_dir()
    try:
        results_dir = scratch / "results"
        results_dir.mkdir()
        manifest_path = results_dir / "baseline_manifest.csv"

        rows = []
        for idx, date in enumerate(["2025-01-02", "2025-01-01"]):
            result_path = results_dir / f"day_{date}.csv"
            pd.DataFrame(
                [
                    {
                        "Day": date,
                        "Sharpe": 1.0 + idx,
                        "Sortino": 2.0 + idx,
                        "Max DD": 0.1 + idx,
                        "P&L-to-MAP": 1.5 + idx,
                        "Final PnL": 2.0 + idx,
                        "Mean |q|": 0.3 + idx,
                        "Near Cap Fraction": 0.05 * idx,
                        "sigma": 0.01 + idx,
                        "A": 0.8 + idx,
                        "kappa": 35000.0 + idx,
                        "as_gamma": 0.1 + idx,
                    }
                ]
            ).to_csv(result_path, index=False)
            rows.append(
                {
                    "day_index": idx,
                    "date": date,
                    "input_path": f"/scratch/input_{idx}.csv",
                    "result_path": str(result_path),
                }
            )

        pd.DataFrame(rows).to_csv(manifest_path, index=False)

        df, summary_path, plot_path = aggregate_manifest_results(manifest_path, pair="DOGEUSDT")

        assert list(df.index) == ["2025-01-01", "2025-01-02"]
        assert summary_path.exists()
        assert plot_path.exists()
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
