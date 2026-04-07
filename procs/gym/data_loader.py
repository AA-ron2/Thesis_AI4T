"""
Data loader for multi-day Tardis L2 order book snapshots.

Usage::

    from procs.gym.data_loader import load_multi_day, load_single_day
    daily_S, daily_dt, dates = load_multi_day(data_dir, pair="DOGEUSDT")
"""

from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd


def load_single_day(filepath: str) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """Load one Tardis CSV → (midprices, dt_array, datetime_index)."""
    data = pd.read_csv(filepath, index_col="timestamp")
    data.index = pd.to_datetime(data.index, unit="us")
    data["midprice"] = (data["asks[0].price"] + data["bids[0].price"]) / 2
    S = data["midprice"].to_numpy()
    ts_ns = data.index.view("int64")
    t_sec = (ts_ns - ts_ns[0]) / 1e9
    dt_sec = np.diff(t_sec, prepend=t_sec[0])
    return S, dt_sec, data.index


def load_multi_day(
    data_dir: str,
    pair: str = "DOGEUSDT",
    pattern: str = "binance_book_snapshot_25_{date}_{pair}.csv",
    max_days: int | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray], list[str]]:
    """
    Load all available days from a directory.

    Returns (daily_midprices, daily_dt_arrays, date_strings).
    """
    glob_pat = pattern.replace("{date}", "*").replace("{pair}", pair)
    files = sorted(glob.glob(os.path.join(data_dir, glob_pat)))

    if not files:
        raise FileNotFoundError(
            f"No files matching '{glob_pat}' in '{data_dir}'."
        )
    if max_days is not None:
        files = files[:max_days]

    daily_S, daily_dt, dates = [], [], []
    for f in files:
        fname = os.path.basename(f)
        # Extract YYYY-MM-DD from filename
        date_str = next(
            (p for p in fname.replace(".csv", "").split("_")
             if len(p) == 10 and p[4] == "-"),
            fname,
        )
        try:
            S, dt, _ = load_single_day(f)
            daily_S.append(S)
            daily_dt.append(dt)
            dates.append(date_str)
            sigma = np.sqrt(np.sum(np.diff(S) ** 2) / max(np.sum(dt[1:]), 1e-10))
            print(f"  {date_str}: {len(S):>8,} snapshots, σ={sigma:.6f}")
        except Exception as e:
            print(f"  SKIP {fname}: {e}")

    print(f"\nLoaded {len(dates)} days.")
    return daily_S, daily_dt, dates
