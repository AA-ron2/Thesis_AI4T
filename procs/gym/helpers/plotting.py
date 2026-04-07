"""
Plotting helpers.

Exact replica of ``procs/gym/helpers/plotting.py`` (Jerome et al., 2023),
adapted for market-replay timestamps.

Functions
---------
plot_trajectory              2×2 panel: cum rewards, prices, inventory+cash, actions
plot_pnl                     Seaborn histogram of episode PnL
generate_results_table_and_hist   Summary table + histogram (for vectorised runs)
get_timestamps               Time axis in seconds
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from procs.agents.agent import Agent
from procs.gym.trading_environment import TradingEnvironment
from procs.gym.index_names import CASH_INDEX, INVENTORY_INDEX, ASSET_PRICE_INDEX
from procs.gym.helpers.generate_trajectory import generate_trajectory


# ═══════════════════════════════════════════════════════════════
# get_timestamps
# ═══════════════════════════════════════════════════════════════
def get_timestamps(env: TradingEnvironment) -> np.ndarray:
    """
    Return a 1-D time axis (in seconds) for an episode.

    For market-replay environments the cumulative sum of the stored
    Δt array gives the real timestamp grid.  For simulated envs this
    falls back to the mbt-gym default ``np.linspace(0, T, n+1)``.

    Returns
    -------
    timestamps : np.ndarray, shape (env.n_steps + 1,)
    """
    mp = env._midprice_model
    if hasattr(mp, "dt_array"):
        return np.cumsum(mp.dt_array)               # real timestamps
    else:
        return np.linspace(0, env.terminal_time, env.n_steps + 1)


# ═══════════════════════════════════════════════════════════════
# plot_trajectory
# ═══════════════════════════════════════════════════════════════
def plot_trajectory(
    env: TradingEnvironment,
    agent: Agent,
    seed: int | None = None,
    datetime_index: np.ndarray | None = None,
) -> None:
    """
    Generate one trajectory and plot the 2×2 diagnostic panel.

    Layout:
        (1,1) cum_rewards     (1,2) asset_prices + bid/ask
        (2,1) inventory+cash  (2,2) actions (δ_bid, δ_ask)

    Parameters
    ----------
    env : TradingEnvironment
    agent : Agent
    seed : int | None
    datetime_index : np.ndarray | None
        If provided (e.g. ``data.index`` for market replay), the
        x-axis shows real timestamps.  If None, uses seconds.
    """
    timestamps = get_timestamps(env)
    observations, actions, rewards = generate_trajectory(env, agent, seed)

    action_dim = actions.shape[1]

    # (num_traj, 1, n_steps) → (num_traj, n_steps)
    rewards_sq = np.squeeze(rewards, axis=1)
    cum_rewards = np.cumsum(rewards_sq, axis=-1)

    cash_holdings = observations[:, CASH_INDEX, :]
    inventory = observations[:, INVENTORY_INDEX, :]
    asset_prices = observations[:, ASSET_PRICE_INDEX, :]

    # Compute bid/ask prices from actions (depths)
    # bid_price = S - δ_bid,  ask_price = S + δ_ask
    midprices_for_actions = asset_prices[:, :-1]   # (N, n_steps)
    bid_prices = midprices_for_actions - actions[:, 0, :]
    ask_prices = midprices_for_actions + actions[:, 1, :]

    # X-axis: real timestamps or seconds
    if datetime_index is not None and len(datetime_index) >= len(timestamps):
        x_full = datetime_index[: len(timestamps)]
        x_steps = datetime_index[: len(timestamps) - 1]
        x_label = "Time"
    else:
        x_full = timestamps
        x_steps = timestamps[:-1]
        x_label = "Time (s)"

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 10))
    ax3a = ax3.twinx()

    for i in range(env.num_trajectories):
        traj_label = f" trajectory {i}" if env.num_trajectories > 1 else ""
        alpha = (i + 1) / (env.num_trajectories + 1)

        # (1,1) Cumulative rewards
        ax1.plot(x_steps, cum_rewards[i, :], alpha=alpha)

        # (1,2) Asset prices + bid/ask
        ax2.plot(x_full, asset_prices[i, :], label="midprice" + traj_label,
                 color="k", alpha=alpha, lw=1.0)
        ax2.plot(x_steps, bid_prices[i, :], label="bid" + traj_label,
                 color="b", alpha=alpha * 0.6, lw=0.7, ls="--")
        ax2.plot(x_steps, ask_prices[i, :], label="ask" + traj_label,
                 color="r", alpha=alpha * 0.6, lw=0.7, ls="--")

        # (2,1) Inventory + cash
        ax3.plot(x_full, inventory[i, :],
                 label="inventory" + traj_label, color="r", alpha=alpha)
        ax3a.plot(x_full, cash_holdings[i, :],
                  label="cash holdings" + traj_label, color="b", alpha=alpha)

        # (2,2) Actions
        ax4.plot(x_steps, actions[i, 0, :],
                 label="δ_bid" + traj_label, color="r", alpha=alpha)
        ax4.plot(x_steps, actions[i, 1, :],
                 label="δ_ask" + traj_label, color="k", alpha=alpha)

    ax1.set_title("cum_rewards")
    ax1.set_ylabel("Cumulative reward")
    ax1.set_xlabel(x_label)

    ax2.set_title("asset_prices")
    ax2.set_ylabel("Price")
    ax2.set_xlabel(x_label)
    ax2.legend(fontsize=8)

    ax3.set_title("inventory and cash holdings")
    ax3.set_ylabel("Inventory (lots)", color="r")
    ax3a.set_ylabel("Cash", color="b")
    ax3.set_xlabel(x_label)
    ax3.legend(loc="upper right", fontsize=8)
    ax3a.legend(loc="upper left", fontsize=8)

    ax4.set_title("Actions (depths)")
    ax4.set_ylabel("Depth (price units)")
    ax4.set_xlabel(x_label)
    ax4.legend(fontsize=8)

    plt.tight_layout()
    plt.show()


# ═══════════════════════════════════════════════════════════════
# plot_pnl
# ═══════════════════════════════════════════════════════════════
def plot_pnl(
    rewards: np.ndarray,
    symmetric_rewards: np.ndarray | None = None,
) -> plt.Figure:
    """
    Seaborn histogram of total-episode rewards.

    Parameters
    ----------
    rewards : np.ndarray, shape (num_trajectories,)
        Total PnL per episode.
    symmetric_rewards : np.ndarray | None
        Optional comparison distribution (e.g. symmetric strategy).

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    if symmetric_rewards is not None:
        sns.histplot(
            symmetric_rewards,
            label="Rewards of symmetric strategy",
            stat="density",
            bins=50,
            ax=ax,
        )
    sns.histplot(
        rewards,
        label="Rewards",
        color="red",
        stat="density",
        bins=50,
        ax=ax,
    )
    ax.legend()
    plt.close()
    return fig


# ═══════════════════════════════════════════════════════════════
# generate_results_table_and_hist
# ═══════════════════════════════════════════════════════════════
def generate_results_table_and_hist(
    vec_env: TradingEnvironment,
    agent: Agent,
    n_episodes: int = 1000,
) -> tuple[pd.DataFrame, plt.Figure, np.ndarray]:
    """
    Run a vectorised back-test and produce a summary table + histogram.

    Same return signature as mbt-gym's ``generate_results_table_and_hist``,
    but uses **streaming statistics** (O(N) memory) instead of storing
    the full O(N × T) observation/action/reward history.

    This fixes the MemoryError for large market-replay datasets
    (e.g. 1000 trajectories × 714k steps = 21.3 GiB).

    Parameters
    ----------
    vec_env : TradingEnvironment
        Environment with ``num_trajectories > 1``.
    agent : Agent
    n_episodes : int
        Not used directly (trajectories come from vec_env.num_trajectories).

    Returns
    -------
    results       : pd.DataFrame   — summary statistics
    fig           : plt.Figure     — PnL histogram
    total_rewards : np.ndarray     — per-trajectory total PnL
    """
    assert vec_env.num_trajectories > 1, (
        "To generate a results table and hist, vec_env must roll out > 1 trajectory."
    )

    from procs.gym.helpers.generate_trajectory_stats import (
        generate_trajectory_stats, stats_to_results_table,
    )

    stats = generate_trajectory_stats(vec_env, agent)
    results = stats_to_results_table(stats)
    total_rewards = stats["total_pnl"]
    fig = plot_pnl(total_rewards)

    return results, fig, total_rewards


# ═══════════════════════════════════════════════════════════════
# plot_learned_policy
# ═══════════════════════════════════════════════════════════════
def plot_learned_policy(
    agent,
    initial_price: float = 100.0,
    terminal_time: float = 1.0,
    n_steps: int = 200,
    inventory_range: list | None = None,
) -> None:
    """
    Plot the learned policy δ_bid(q, τ) and δ_ask(q, τ) as a function
    of time for different inventory levels.

    Expected A-S patterns:
        • δ_bid increases with q (long → passive buy)
        • δ_ask decreases with q (long → aggressive sell)
        • Both decrease as τ → 0 (less inventory risk near terminal)

    Parameters
    ----------
    agent : Agent
        Must support ``agent.get_action(raw_obs)``.
    initial_price : float
    terminal_time : float
    n_steps : int
    inventory_range : list
        Inventory values to plot.  Default [-3, -2, -1, 0, 1, 2, 3].
    """
    if inventory_range is None:
        inventory_range = [-3, -2, -1, 0, 1, 2, 3]

    timestamps = np.linspace(0, terminal_time, n_steps + 1)[:-1]
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(inventory_range)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

    for idx, q_val in enumerate(inventory_range):
        bid_depths, ask_depths = [], []
        for t_val in timestamps:
            raw_obs = np.array([[0.0, q_val, t_val, initial_price]])
            action = agent.get_action(raw_obs)
            bid_depths.append(action[0, 0])
            ask_depths.append(action[0, 1])

        ax1.plot(timestamps, bid_depths, label=f"q={q_val}", color=colors[idx])
        ax2.plot(timestamps, ask_depths, label=f"q={q_val}", color=colors[idx])

    ax1.set_title("Bid depth (δ_bid) — learned policy")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("δ_bid (price units)")
    ax1.legend()
    ax1.grid(True)

    ax2.set_title("Ask depth (δ_ask) — learned policy")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("δ_ask (price units)")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


# ═══════════════════════════════════════════════════════════════
# plot_cvar_training
# ═══════════════════════════════════════════════════════════════
def plot_cvar_training(
    cvar_history: list[float],
    lambda_history: list[float],
    dd_threshold: float,
) -> None:
    """
    Plot CVaR and Lagrange multiplier λ convergence during training.

    Parameters
    ----------
    cvar_history : list[float]
        CVaR_α(max_drawdown) at each PPO rollout.
    lambda_history : list[float]
        λ at each PPO rollout.
    dd_threshold : float
        The CVaR constraint threshold d_max.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    iters = range(1, len(cvar_history) + 1)

    ax1.plot(iters, cvar_history, "b-", lw=1.5, label="CVaR_α(DD)")
    ax1.axhline(y=dd_threshold, color="r", ls="--", lw=1.0,
                label=f"threshold d={dd_threshold}")
    ax1.set_xlabel("PPO rollout iteration")
    ax1.set_ylabel("CVaR_α(max drawdown)")
    ax1.set_title("CVaR Convergence")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(iters, lambda_history, "g-", lw=1.5)
    ax2.set_xlabel("PPO rollout iteration")
    ax2.set_ylabel("λ (Lagrange multiplier)")
    ax2.set_title("Lagrange Multiplier λ")
    ax2.grid(True)

    plt.tight_layout()
    plt.show()
