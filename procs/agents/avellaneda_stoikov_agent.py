"""
Avellaneda-Stoikov analytical agent.

Computes the optimal bid/ask depths from the closed-form solution
in Avellaneda & Stoikov (2008), §2.4 (infinite-horizon / stationary
version):

    Reservation price:
        r(s, q)  =  s  −  q · γ · σ² · τ

    Optimal spread:
        δ*(τ)    =  γ · σ² · τ  +  (2 / γ) · ln(1 + γ / κ)

    Optimal quotes:
        p_bid  =  r  −  δ* / 2
        p_ask  =  r  +  δ* / 2

    Action (depths):
        δ_bid  =  s − p_bid  =   q · γ · σ² · τ  +  δ* / 2
        δ_ask  =  p_ask − s  =  −q · γ · σ² · τ  +  δ* / 2

Architecture reference:
    Matches the ``Agent`` base + ``AvellanedaStoikovAgent`` in mbt-gym
    (Jerome et al., 2023, §4.1).

Tick snapping:
    After computing ideal bid/ask *prices*, they are snapped to the tick
    grid (floor for bid, ceil for ask) before converting to depths.
    This is a practical detail from the original monolithic code.
"""

from __future__ import annotations

import numpy as np

from procs.gym.index_names import (
    INVENTORY_INDEX, TIME_INDEX, ASSET_PRICE_INDEX,
)
from procs.agents.agent import Agent


# ── concrete: Avellaneda-Stoikov ──────────────────────────────
class AvellanedaStoikovAgent(Agent):
    """
    Analytical Avellaneda-Stoikov market-making agent.

    Parameters
    ----------
    gamma : float
        Risk-aversion parameter γ.
    sigma : float
        Midprice volatility σ in price / √second  (arithmetic BM).
    kappa : float
        Fill-decay parameter κ in USDT⁻¹.
    terminal_time : float
        Session duration T in seconds.
    tick_size : float
        Minimum price increment for DOGEUSDT  (0.00001).
    """

    def __init__(
        self,
        gamma: float,
        sigma: float,
        kappa: float,
        terminal_time: float,
        tick_size: float = 0.00001,
    ):
        self.gamma = gamma
        self.sigma = sigma
        self.kappa = kappa
        self.terminal_time = terminal_time
        self.tick_size = tick_size

    # ── public interface ──────────────────────────────────────
    def get_action(self, state: np.ndarray) -> np.ndarray:
        """
        Compute optimal bid/ask depths.

        Parameters
        ----------
        state : np.ndarray, shape (num_trajectories, 4)
            Observation matrix [cash, inventory, time, midprice].

        Returns
        -------
        action : np.ndarray, shape (num_trajectories, 2)
            [δ_bid, δ_ask]  in price units (USDT), ≥ 0.
        """
        # Accept both 1-D and 2-D input
        if state.ndim == 1:
            state = state.reshape(1, -1)

        s = state[:, ASSET_PRICE_INDEX]                 # (N,)
        q = state[:, INVENTORY_INDEX]                   # (N,)
        t = state[:, TIME_INDEX]                        # (N,)
        tau = np.maximum(self.terminal_time - t, 0.0)   # (N,)

        # Reservation price  (A-S eq. 8)
        reservation = s - q * self.gamma * self.sigma ** 2 * tau

        # Optimal spread  (A-S §2.4 / eq. 10)
        spread = (
            self.gamma * self.sigma ** 2 * tau
            + (2.0 / self.gamma) * np.log(1.0 + self.gamma / self.kappa)
        )

        # Ideal quotes (continuous)
        bid_ideal = reservation - 0.5 * spread
        ask_ideal = reservation + 0.5 * spread

        # Snap to tick grid
        bid = np.floor(bid_ideal / self.tick_size) * self.tick_size
        ask = np.ceil(ask_ideal / self.tick_size) * self.tick_size

        # Fix crossed quotes
        crossed = ask <= bid
        ask = np.where(crossed, bid + self.tick_size, ask)

        # Convert to depths  (non-negative)
        min_delta = 0.5 * self.tick_size
        delta_bid = np.maximum(s - bid, min_delta)
        delta_ask = np.maximum(ask - s, min_delta)

        return np.column_stack([delta_bid, delta_ask])  # (N, 2)

    # ── diagnostics (not called by env, useful in notebooks) ──
    def reservation_price(self, s: float, q: float, t: float) -> float:
        tau = max(self.terminal_time - t, 0.0)
        return s - q * self.gamma * self.sigma ** 2 * tau

    def optimal_spread(self, t: float) -> float:
        tau = max(self.terminal_time - t, 0.0)
        return (
            self.gamma * self.sigma ** 2 * tau
            + (2.0 / self.gamma) * np.log(1.0 + self.gamma / self.kappa)
        )


class AvellanedaStoikovInfiniteHorizonAgent(Agent):
    """
    Infinite-horizon (stationary) Avellaneda-Stoikov agent.

    Reservation price (A-S §2.3):
        r̃(s, q) = s + (1/γ) · ln(1 + (1−2q)γ²σ² / (2ω − γ²σ²q²))

    where ω > (1/2)γ²σ²q² ensures bounded prices.
    Natural choice: ω = (1/2)γ²σ²(q_max + 1)².

    Optimal spread (A-S §2.3):
        δ̃ᵃ = r̃ᵃ − s = (1/γ)·ln(1 + (1−2q)γ²σ²/(2ω−γ²σ²q²))
                          + (1/γ)·ln(1 + γ/κ)
        δ̃ᵇ = s − r̃ᵇ   (symmetric form)

    The spread is TIME-INDEPENDENT — no τ term.
    This is the key difference from the finite-horizon version.

    Reference: Avellaneda & Stoikov (2008), §2.3, eqs. (9)–(10).
    """

    def __init__(
        self,
        gamma: float,
        sigma: float,
        kappa: float,
        q_max: int = 50,
        tick_size: float = 0.00001,
    ):
        self.gamma = gamma
        self.sigma = sigma
        self.kappa = kappa
        self.q_max = q_max
        self.tick_size = tick_size

        # ω parameter (A-S §2.3): ensures prices stay bounded
        self.omega = 0.5 * gamma**2 * sigma**2 * (q_max + 1)**2

    def get_action(self, state: np.ndarray) -> np.ndarray:
        if state.ndim == 1:
            state = state.reshape(1, -1)

        s = state[:, 3]   # ASSET_PRICE_INDEX
        q = state[:, 1]   # INVENTORY_INDEX

        g = self.gamma
        g2s2 = g**2 * self.sigma**2

        # Reservation price adjustment (A-S eq. 9, infinite horizon)
        denom = 2.0 * self.omega - g2s2 * q**2
        # Safety: clamp denominator away from zero
        denom = np.maximum(denom, 1e-10)

        reservation = s + (1.0 / g) * np.log(1.0 + (1.0 - 2.0 * q) * g2s2 / denom)

        # Spread component from fill probability (same as finite horizon)
        spread_fill = (1.0 / g) * np.log(1.0 + g / self.kappa)

        # Bid/ask prices
        bid_ideal = reservation - spread_fill
        ask_ideal = reservation + spread_fill

        # Snap to tick grid
        bid = np.floor(bid_ideal / self.tick_size) * self.tick_size
        ask = np.ceil(ask_ideal / self.tick_size) * self.tick_size
        crossed = ask <= bid
        ask = np.where(crossed, bid + self.tick_size, ask)

        min_delta = 0.5 * self.tick_size
        delta_bid = np.maximum(s - bid, min_delta)
        delta_ask = np.maximum(ask - s, min_delta)

        return np.column_stack([delta_bid, delta_ask])

    def reservation_price(self, s: float, q: float) -> float:
        g2s2 = self.gamma**2 * self.sigma**2
        denom = max(2.0 * self.omega - g2s2 * q**2, 1e-10)
        return s + (1.0 / self.gamma) * np.log(1.0 + (1.0 - 2.0 * q) * g2s2 / denom)

    def optimal_spread(self) -> float:
        """Time-independent spread (the defining feature of infinite horizon)."""
        return (2.0 / self.gamma) * np.log(1.0 + self.gamma / self.kappa)