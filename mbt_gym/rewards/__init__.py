"""
Reward functions.

This module provides:

    • ``PnLReward`` — change in mark-to-market portfolio value.
      Used for A-S baseline evaluation.

    • ``CjMmCriterion`` — Cartea-Jaimungal market-making criterion:
      PnL minus a running inventory penalty.
      Used for RL training (gives the agent a clear gradient signal
      to manage inventory rather than avoid trading).

References:
    • Jerome et al. (2023), §3 — mbt-gym reward function design
    • Cartea, Jaimungal & Penalva (2015), §10.3 — inventory aversion
"""

from __future__ import annotations

import numpy as np

# Inline constants to avoid circular import through gym/__init__
# (same values as mbt_gym.gym.index_names)
CASH_INDEX = 0
INVENTORY_INDEX = 1
TIME_INDEX = 2
ASSET_PRICE_INDEX = 3


# ── base class ────────────────────────────────────────────────
class RewardFunction:
    """Minimal base following mbt-gym's ``RewardFunction``."""

    def calculate(
        self,
        current_state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        is_terminal: bool = False,
    ) -> np.ndarray:
        raise NotImplementedError

    def reset(self, initial_state: np.ndarray) -> None:
        pass


# ── concrete: mark-to-market PnL ──────────────────────────────
class PnLReward(RewardFunction):
    """
    Per-step change in mark-to-market portfolio value.

    V_t  = cash_t + q_t · S_t
    r_t  = V_t  − V_{t-1}

    Matches mbt-gym's ``PnL`` reward (Jerome et al. 2023, §3).
    """

    def calculate(
        self,
        current_state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        is_terminal: bool = False,
    ) -> np.ndarray:
        def _mtm(state: np.ndarray) -> np.ndarray:
            return (
                state[:, CASH_INDEX]
                + state[:, INVENTORY_INDEX] * state[:, ASSET_PRICE_INDEX]
            )
        return _mtm(next_state) - _mtm(current_state)


# ── concrete: Cartea-Jaimungal market-making criterion ────────
class CjMmCriterion(RewardFunction):
    """
    Cartea-Jaimungal market-making reward with running inventory penalty.

        r_t = ΔPnL  −  φ · q² · Δt

    where φ is ``per_step_inventory_aversion``.

    This is the standard reward for RL market-making training:
    the ``−φq²Δt`` term gives the agent a *direct gradient signal*
    to manage inventory.  Without it, PPO learns to avoid trading
    entirely (PnL is dominated by uncontrollable midprice noise).

    Matches mbt-gym's ``CjMmCriterion`` (Jerome et al. 2023, §3).

    Parameters
    ----------
    per_step_inventory_aversion : float
        φ — penalty per unit of q² per unit time.
        Typical values: 0.001–0.01 for BM envs, scale by σ² for real data.
    terminal_inventory_aversion : float
        Additional terminal penalty weight.  Usually 0 (handled by φ).
    inventory_exponent : float
        Exponent on |q|.  Default 2 (quadratic).
    """

    def __init__(
        self,
        per_step_inventory_aversion: float = 0.01,
        terminal_inventory_aversion: float = 0.0,
        inventory_exponent: float = 2.0,
    ):
        self.per_step_inventory_aversion = per_step_inventory_aversion
        self.terminal_inventory_aversion = terminal_inventory_aversion
        self.inventory_exponent = inventory_exponent
        self.pnl = PnLReward()

    def calculate(
        self,
        current_state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        is_terminal: bool = False,
    ) -> np.ndarray:
        dt = next_state[:, TIME_INDEX] - current_state[:, TIME_INDEX]
        pnl_change = self.pnl.calculate(current_state, action, next_state, is_terminal)
        inventory_penalty = (
            dt * self.per_step_inventory_aversion
            * np.abs(next_state[:, INVENTORY_INDEX]) ** self.inventory_exponent
        )
        return pnl_change - inventory_penalty

class CjMmDrawdownPenalty(RewardFunction):
    """
    CjMmCriterion with an additional running drawdown penalty:

        r_t = ΔPnL − φ·q²·Δt − α·drawdown_t

    where drawdown_t = max(peak_PnL − current_PnL, 0).

    The penalty weight α is fixed — this is the "penalise drawdown"
    baseline against which the CVaR-constrained agent is compared.

    References:
        • Moody & Saffell (2001), IEEE Trans. Neural Networks, 12(4)
        • Falces Marin et al. (2022), §7 (suggested, not implemented)
    """

    def __init__(
        self,
        per_step_inventory_aversion: float = 0.01,
        drawdown_penalty: float = 1.0,
        inventory_exponent: float = 2.0,
    ):
        self.per_step_inventory_aversion = per_step_inventory_aversion
        self.drawdown_penalty = drawdown_penalty
        self.inventory_exponent = inventory_exponent
        self.pnl = PnLReward()
        self._peak_pnl = None

    def calculate(
        self,
        current_state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        is_terminal: bool = False,
    ) -> np.ndarray:
        dt = next_state[:, TIME_INDEX] - current_state[:, TIME_INDEX]
        pnl_change = self.pnl.calculate(current_state, action, next_state, is_terminal)

        # Inventory penalty (same as CjMmCriterion)
        inv_penalty = (
            dt * self.per_step_inventory_aversion
            * np.abs(next_state[:, INVENTORY_INDEX]) ** self.inventory_exponent
        )

        # Running drawdown
        current_pnl = (
            next_state[:, CASH_INDEX]
            + next_state[:, INVENTORY_INDEX] * next_state[:, ASSET_PRICE_INDEX]
        )
        self._peak_pnl = np.maximum(self._peak_pnl, current_pnl)
        drawdown = self._peak_pnl - current_pnl              # ≥ 0

        return pnl_change - inv_penalty - self.drawdown_penalty * drawdown

    def reset(self, initial_state: np.ndarray) -> None:
        self._peak_pnl = (
            initial_state[:, CASH_INDEX]
            + initial_state[:, INVENTORY_INDEX] * initial_state[:, ASSET_PRICE_INDEX]
        )
