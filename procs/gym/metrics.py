"""
Performance metrics for market-making back-tests.

Implements the four indicators from Falces Marin et al. (2022),
"A reinforcement learning approach to improve the performance of the
Avellaneda-Stoikov market-making algorithm", *PLOS ONE* 17(12), §5.3:

1. **Sharpe ratio**  — risk-adjusted return  (eq. 16)
       Sharpe  =  mean(returns) / std(returns)

2. **Sortino ratio**  — downside-risk-adjusted return  (eq. 20)
       Sortino =  mean(returns) / std(negative returns)

3. **Maximum drawdown**  — largest peak-to-trough drop in portfolio
   value during the episode  (§5.3, referencing Gašperov [25])

4. **P&L-to-MAP**  — final open PnL divided by mean absolute inventory
   position  (eq. 21)
       P&L-to-MAP  =  Ψ(T) / mean(|I|)

All ratios are computed from **Close P&L returns** (per-step ΔPnL)
except P&L-to-MAP which uses the **Open PnL** — matching the paper's
methodology (§6, first paragraph).

Additionally provides ``backtest_summary`` which runs vectorised
trajectories and produces a DataFrame with all four metrics per
trajectory, plus aggregate statistics.

Architecture reference:
    Extends ``procs/gym/backtesting.py`` (Jerome et al., 2023)
    with the Falces Marin indicator set.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from procs.gym.trading_environment import TradingEnvironment
from procs.agents.agent import Agent
from procs.gym.index_names import CASH_INDEX, INVENTORY_INDEX, ASSET_PRICE_INDEX


# ═══════════════════════════════════════════════════════════════
# Core metric functions  (operate on pre-computed arrays)
# ═══════════════════════════════════════════════════════════════

def sharpe_ratio(returns: np.ndarray) -> float:
    """
    Sharpe ratio  =  mean(r) / std(r).

    Falces Marin et al. (2022), eq. 16.  No annualisation —
    computed over the full episode, same as the paper.

    Parameters
    ----------
    returns : np.ndarray, shape (T,)
        Per-step PnL changes (close PnL returns).
    """
    std = returns.std()
    if std < 1e-15:
        return 0.0
    return float(returns.mean() / std)


def sortino_ratio(returns: np.ndarray) -> float:
    """
    Sortino ratio  =  mean(r) / std(negative r).

    Falces Marin et al. (2022), eq. 20.

    Parameters
    ----------
    returns : np.ndarray, shape (T,)
        Per-step PnL changes (close PnL returns).
    """
    neg = returns[returns < 0]
    if len(neg) < 2:
        return 0.0
    std_neg = neg.std()
    if std_neg < 1e-15:
        return 0.0
    return float(returns.mean() / std_neg)


def maximum_drawdown(pnl_path: np.ndarray) -> float:
    """
    Maximum drawdown  =  max(peak − trough) over the PnL path.

    Falces Marin et al. (2022), §5.3; Gašperov & Kostanjčar [25].

    Parameters
    ----------
    pnl_path : np.ndarray, shape (T+1,)
        Cumulative mark-to-market PnL  (cash + q·S at each step).

    Returns
    -------
    max_dd : float
        Maximum drawdown (positive number; 0 if PnL never drops).
    """
    running_max = np.maximum.accumulate(pnl_path)
    drawdowns = running_max - pnl_path
    return float(drawdowns.max())


def pnl_to_map(final_pnl: float, inventory_path: np.ndarray) -> float:
    """
    P&L to Mean Absolute Position  =  Ψ(T) / mean(|I|).

    Falces Marin et al. (2022), eq. 21; Gašperov & Kostanjčar [25].

    Parameters
    ----------
    final_pnl : float
        Open P&L at terminal time  Ψ(T).
    inventory_path : np.ndarray, shape (T+1,)
        Inventory time-series  I(t).

    Returns
    -------
    ratio : float
        P&L-to-MAP.  Returns 0 if mean(|I|) ≈ 0 (no trading).
    """
    mean_abs_pos = np.abs(inventory_path).mean()
    if mean_abs_pos < 1e-15:
        return 0.0
    return float(final_pnl / mean_abs_pos)


# ═══════════════════════════════════════════════════════════════
# Convenience wrappers — single run, all metrics at once
# ═══════════════════════════════════════════════════════════════

def get_all_metrics(
    env: TradingEnvironment, agent: Agent, seed: int | None = None,
) -> dict[str, float]:
    """
    Run **one** trajectory and compute all four Falces Marin metrics.

    This replaces the old pattern of calling ``get_sharpe_ratio``,
    ``get_sortino_ratio``, ``get_maximum_drawdown``, ``get_pnl_to_map``
    separately (which ran the environment four times).

    Returns
    -------
    dict with keys: sharpe, sortino, max_drawdown, pnl_to_map, total_pnl
    """
    from procs.gym.helpers.generate_trajectory_stats import generate_trajectory_stats

    stats = generate_trajectory_stats(env, agent, seed=seed)
    return {
        "sharpe":       float(stats["sharpe"][0]),
        "sortino":      float(stats["sortino"][0]),
        "max_drawdown": float(stats["max_drawdown"][0]),
        "pnl_to_map":   float(stats["pnl_to_map"][0]),
        "total_pnl":    float(stats["total_pnl"][0]),
        "terminal_q":   float(stats["terminal_q"][0]),
    }


def get_sharpe_ratio(
    env: TradingEnvironment, agent: Agent, seed: int | None = None,
) -> float:
    """Run one trajectory and compute the Sharpe ratio."""
    return get_all_metrics(env, agent, seed)["sharpe"]


def get_sortino_ratio(
    env: TradingEnvironment, agent: Agent, seed: int | None = None,
) -> float:
    """Run one trajectory and compute the Sortino ratio."""
    return get_all_metrics(env, agent, seed)["sortino"]


def get_maximum_drawdown(
    env: TradingEnvironment, agent: Agent, seed: int | None = None,
) -> float:
    """Run one trajectory and compute the maximum drawdown."""
    return get_all_metrics(env, agent, seed)["max_drawdown"]


def get_pnl_to_map(
    env: TradingEnvironment, agent: Agent, seed: int | None = None,
) -> float:
    """Run one trajectory and compute P&L-to-MAP."""
    return get_all_metrics(env, agent, seed)["pnl_to_map"]


# ═══════════════════════════════════════════════════════════════
# Vectorised summary  (batch all 4 metrics over N trajectories)
# ═══════════════════════════════════════════════════════════════

def backtest_summary(
    env: TradingEnvironment,
    agent: Agent,
    seed: int | None = None,
) -> pd.DataFrame:
    """
    Run N = ``env.num_trajectories`` episodes and compute all four
    Falces Marin indicators, using **streaming statistics** (O(N) memory).

    Parameters
    ----------
    env : TradingEnvironment
    agent : Agent
    seed : int | None

    Returns
    -------
    summary : pd.DataFrame
        Columns = the four indicators + final PnL + terminal inventory.
        One row per trajectory, plus Mean / Std / Median footer rows.
    """
    from procs.gym.helpers.generate_trajectory_stats import (
        generate_trajectory_stats, stats_to_summary,
    )
    stats = generate_trajectory_stats(env, agent, seed=seed)
    return stats_to_summary(stats)
