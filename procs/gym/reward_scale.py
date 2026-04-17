"""
Reward scale estimation.

Estimates the typical per-episode reward magnitude by running a
fixed-spread agent.  The inverse of this value is used as
``reward_scale`` to normalise rewards to O(1) for RL training.

This matches the logic in mbt-gym's
``TradingEnvironment._get_inventory_neutral_rewards()``.
"""

from __future__ import annotations

import numpy as np

from procs.gym.helpers.fast_rollout import fast_simulate


def estimate_reward_scale(
    midprices: np.ndarray | None = None,
    dt_array: np.ndarray | None = None,
    sigma: float = 2.0,
    kappa: float = 1.5,
    A: float = 140.0,
    terminal_time: float = 1.0,
    n_steps: int = 200,
    tick_size: float = 0.01,
    Q_MAX: int = 10,
    num_trajectories: int = 1000,
    use_bm: bool = True,
) -> float:
    """
    Estimate the reward scale for RL training.

    Runs a near-optimal fixed-spread agent (δ = 1/κ on both sides)
    and computes the mean total PnL.  The reward scale is ``1 / mean_pnl``.

    Parameters
    ----------
    midprices, dt_array : np.ndarray | None
        If provided, uses market replay.
    sigma, kappa, A, terminal_time, n_steps : float
        BM model parameters (used if ``use_bm=True``).
    use_bm : bool
        If True, generates BM midprice.  If False, uses market replay.
    num_trajectories : int
        N for averaging.

    Returns
    -------
    scale : float
        Multiply per-step rewards by this to get O(1) magnitude.
    """
    if use_bm:
        from procs.stochastic_processes.midprice_models import BrownianMotionMidpriceModel
        bm = BrownianMotionMidpriceModel(
            volatility=sigma, initial_price=100.0,
            terminal_time=terminal_time, n_steps=n_steps,
            num_trajectories=1,
        )
        midprices_gen = np.zeros(n_steps + 1)
        midprices_gen[0] = 100.0
        dt_val = terminal_time / n_steps
        rng = np.random.default_rng(0)
        for i in range(1, n_steps + 1):
            midprices_gen[i] = midprices_gen[i-1] + sigma * np.sqrt(dt_val) * rng.normal()
        dt_arr = np.full(n_steps + 1, dt_val)
        dt_arr[0] = 0.0
        midprices = midprices_gen
        dt_array = dt_arr
        T = terminal_time
    else:
        T = float(dt_array.sum())

    # Fixed-spread agent: δ = 1/κ (approximately optimal for risk-neutral)
    gamma_neutral = 0.001  # near risk-neutral
    stats = fast_simulate(
        midprices=midprices,
        dt_array=dt_array,
        gamma=gamma_neutral,
        sigma=sigma,
        kappa=kappa,
        A=A,
        terminal_time=T,
        tick_size=tick_size,
        Q_MAX=Q_MAX,
        num_trajectories=num_trajectories,
        seed=0,
        use_linear_approximation=True if use_bm else False,
    )

    mean_pnl = np.abs(stats["total_pnl"]).mean()
    if mean_pnl < 1e-15:
        return 1.0

    # Scale so that mean episode reward ≈ 1.0
    # Per-step scale = 1 / mean_total_pnl (since reward_scale * Σr ≈ 1)
    return float(1.0 / mean_pnl)
