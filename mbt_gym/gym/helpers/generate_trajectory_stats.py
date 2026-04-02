"""
Streaming trajectory statistics.

Runs the environment once and computes all Falces Marin et al. (2022)
metrics **inline** during the rollout, using O(N) memory instead of
storing the full O(N × T) observation/action/reward history.

This solves two problems:
    1. **Memory**: 1000 trajectories × 714k steps × 4 features × 8 bytes
       = 21.3 GiB with full storage.  With streaming: ~40 KB total.
    2. **Speed**: metrics are computed from a single rollout instead of
       re-running the environment four times.

Running statistics tracked per trajectory (all shape (N,)):
    - Welford accumulators for Sharpe:  Σr, Σr², count
    - Welford accumulators for Sortino: Σr_neg, Σr²_neg, count_neg
    - Max drawdown: running_max_pnl, max_drawdown
    - P&L-to-MAP: Σ|q|, count
    - Spread: Σ(δ_bid + δ_ask), count
    - Terminal state: final cash, inventory, midprice, PnL
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mbt_gym.gym.trading_environment import TradingEnvironment
from mbt_gym.agents.agent import Agent
from mbt_gym.gym.index_names import CASH_INDEX, INVENTORY_INDEX, ASSET_PRICE_INDEX


def generate_trajectory_stats(
    env: TradingEnvironment,
    agent: Agent,
    seed: int | None = None,
) -> dict[str, np.ndarray]:
    """
    Run one full episode and compute per-trajectory statistics inline.

    Parameters
    ----------
    env : TradingEnvironment
    agent : Agent
    seed : int | None

    Returns
    -------
    stats : dict[str, np.ndarray]
        All values have shape ``(num_trajectories,)``:

        - ``total_pnl``        : final mark-to-market PnL
        - ``terminal_q``       : terminal inventory
        - ``mean_spread``      : time-averaged bid-ask spread
        - ``sharpe``           : Sharpe ratio  (Falces Marin eq. 16)
        - ``sortino``          : Sortino ratio (Falces Marin eq. 20)
        - ``max_drawdown``     : Maximum drawdown (§5.3)
        - ``pnl_to_map``       : P&L / mean(|q|) (Falces Marin eq. 21)
        - ``n_steps``          : number of steps executed (scalar)
    """
    if seed is not None:
        env.seed(seed)

    N = env.num_trajectories
    obs, _ = env.reset()

    # ── Running accumulators (all shape (N,)) ─────────────────
    # PnL path
    pnl = (
        obs[:, CASH_INDEX]
        + obs[:, INVENTORY_INDEX] * obs[:, ASSET_PRICE_INDEX]
    )
    running_max_pnl = pnl.copy()
    max_drawdown = np.zeros(N)

    # Welford accumulators for returns
    sum_r = np.zeros(N)
    sum_r2 = np.zeros(N)
    sum_r_neg = np.zeros(N)
    sum_r2_neg = np.zeros(N)
    count_neg = np.zeros(N)

    # Inventory and spread
    sum_abs_q = np.abs(obs[:, INVENTORY_INDEX]).copy()
    sum_spread = np.zeros(N)
    step_count = 0

    # ── Main loop ─────────────────────────────────────────────
    while True:
        action = agent.get_action(obs)
        if action.ndim == 1:
            action = np.repeat(action.reshape(1, -1), N, axis=0)

        obs, reward, done, truncated, info = env.step(action)
        step_count += 1

        r = reward  # (N,)

        # Sharpe accumulators
        sum_r += r
        sum_r2 += r * r

        # Sortino accumulators
        neg_mask = r < 0
        sum_r_neg += np.where(neg_mask, r, 0.0)
        sum_r2_neg += np.where(neg_mask, r * r, 0.0)
        count_neg += neg_mask.astype(np.float64)

        # PnL path → drawdown
        pnl = (
            obs[:, CASH_INDEX]
            + obs[:, INVENTORY_INDEX] * obs[:, ASSET_PRICE_INDEX]
        )
        running_max_pnl = np.maximum(running_max_pnl, pnl)
        max_drawdown = np.maximum(max_drawdown, running_max_pnl - pnl)

        # Inventory and spread
        sum_abs_q += np.abs(obs[:, INVENTORY_INDEX])
        sum_spread += action[:, 0] + action[:, 1]

        if done[0]:
            break

    # ── Compute final metrics ─────────────────────────────────
    T = step_count
    obs_count = T + 1  # includes initial observation

    mean_r = sum_r / T
    var_r = sum_r2 / T - mean_r ** 2
    std_r = np.sqrt(np.maximum(var_r, 0.0))

    # Sortino: std of negative returns only
    safe_count_neg = np.maximum(count_neg, 2.0)
    mean_r_neg = sum_r_neg / safe_count_neg
    var_r_neg = sum_r2_neg / safe_count_neg - mean_r_neg ** 2
    std_r_neg = np.sqrt(np.maximum(var_r_neg, 0.0))

    sharpe = np.where(std_r > 1e-15, mean_r / std_r, 0.0)
    sortino = np.where(
        (std_r_neg > 1e-15) & (count_neg >= 2),
        mean_r / std_r_neg,
        0.0,
    )

    mean_abs_q = sum_abs_q / obs_count
    pnl_to_map = np.where(mean_abs_q > 1e-15, pnl / mean_abs_q, 0.0)

    return {
        "total_pnl": pnl,
        "terminal_q": obs[:, INVENTORY_INDEX],
        "mean_spread": sum_spread / T,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_drawdown,
        "pnl_to_map": pnl_to_map,
        "mean_abs_q": mean_abs_q,
        "n_steps": T,
    }


def stats_to_summary(stats: dict[str, np.ndarray]) -> pd.DataFrame:
    """
    Convert streaming stats dict into a Falces Marin summary DataFrame.

    Same output format as ``backtest_summary`` but computed from
    the O(N)-memory streaming stats.
    """
    df = pd.DataFrame({
        "Sharpe": stats["sharpe"],
        "Sortino": stats["sortino"],
        "Max DD": stats["max_drawdown"],
        "P&L-to-MAP": stats["pnl_to_map"],
        "Final PnL": stats["total_pnl"],
        "Terminal q": stats["terminal_q"],
        "Mean |q|": stats["mean_abs_q"],
        "Mean spread": stats["mean_spread"],
    })
    N = len(df)
    df.index = [f"traj_{i}" for i in range(N)]
    df.index.name = "trajectory"

    agg = pd.DataFrame({
        col: [df[col].mean(), df[col].std(), df[col].median()]
        for col in df.columns
    }, index=["Mean", "Std", "Median"])
    agg.index.name = "trajectory"

    return pd.concat([df, agg])


def stats_to_results_table(stats: dict[str, np.ndarray]) -> pd.DataFrame:
    """
    Convert streaming stats into the mbt-gym ``generate_results_table_and_hist``
    DataFrame format.
    """
    pnl = stats["total_pnl"]
    q = stats["terminal_q"]
    spread = stats["mean_spread"]

    results = pd.DataFrame(
        index=["Inventory"],
        columns=[
            "Mean spread", "Mean PnL", "Std PnL",
            "Mean terminal inventory", "Std terminal inventory",
        ],
    )
    results.loc["Inventory", "Mean spread"] = float(np.mean(spread))
    results.loc["Inventory", "Mean PnL"] = float(np.mean(pnl))
    results.loc["Inventory", "Std PnL"] = float(np.std(pnl))
    results.loc["Inventory", "Mean terminal inventory"] = float(np.mean(q))
    results.loc["Inventory", "Std terminal inventory"] = float(np.std(q))
    return results
