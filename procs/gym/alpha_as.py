from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd

try:  # pragma: no cover - depends on runtime environment
    import gymnasium as gym
    from gymnasium import spaces
    _GYMNASIUM_AVAILABLE = True
except Exception:  # pragma: no cover - imported in lightweight/static checks
    gym = None
    spaces = None
    _GYMNASIUM_AVAILABLE = False


DEFAULT_ALPHA_AS_GAMMAS = (0.01, 0.1, 0.2, 0.9)
DEFAULT_ALPHA_AS_SKEWS = (-0.1, -0.05, 0.0, 0.05, 0.1)


@dataclass(frozen=True)
class AlphaASAction:
    index: int
    gamma: float
    skew: float


@dataclass(frozen=True)
class AlphaASQuote:
    bid: float
    ask: float
    delta_bid: float
    delta_ask: float


def build_alpha_as_action_grid(
    gammas: Iterable[float] = DEFAULT_ALPHA_AS_GAMMAS,
    skews: Iterable[float] = DEFAULT_ALPHA_AS_SKEWS,
) -> tuple[AlphaASAction, ...]:
    """Return the 4 x 5 discrete Alpha-AS action grid."""
    actions: list[AlphaASAction] = []
    for gamma in gammas:
        for skew in skews:
            actions.append(AlphaASAction(len(actions), float(gamma), float(skew)))
    return tuple(actions)


def decode_alpha_as_action(
    action_index: int,
    grid: tuple[AlphaASAction, ...] | None = None,
) -> AlphaASAction:
    """Map a DQN action id to its ``(gamma, skew)`` pair."""
    action_grid = grid or build_alpha_as_action_grid()
    if action_index < 0 or action_index >= len(action_grid):
        raise IndexError(f"Alpha-AS action index {action_index} outside 0..{len(action_grid)-1}.")
    return action_grid[int(action_index)]


def compute_alpha_as_quote(
    *,
    midprice: float,
    inventory: float,
    time_elapsed: float,
    gamma: float,
    skew: float,
    sigma: float,
    kappa: float,
    terminal_time: float,
    tick_size: float,
) -> AlphaASQuote:
    """Compute A-S quotes after applying an Alpha-AS multiplicative skew."""
    tau = max(float(terminal_time) - float(time_elapsed), 0.0)
    mid = float(midprice)
    gamma = float(gamma)
    sigma = float(sigma)
    kappa = float(kappa)

    reservation = mid - float(inventory) * gamma * sigma**2 * tau
    spread = gamma * sigma**2 * tau + (2.0 / gamma) * np.log(1.0 + gamma / kappa)

    bid_ideal = reservation - 0.5 * spread
    ask_ideal = reservation + 0.5 * spread

    # Falces Marin et al. skew the AS output prices.  With crypto prices near
    # zero, a literal 10% shift can move one quote through the mid, so clamp
    # after skewing to keep executable depths nonnegative.
    skew_multiplier = 1.0 + float(skew)
    bid_ideal *= skew_multiplier
    ask_ideal *= skew_multiplier

    bid = np.floor(bid_ideal / tick_size) * tick_size
    ask = np.ceil(ask_ideal / tick_size) * tick_size

    min_delta = 0.5 * tick_size
    bid = min(bid, mid - min_delta)
    ask = max(ask, mid + min_delta)
    if ask <= bid:
        ask = bid + tick_size

    return AlphaASQuote(
        bid=float(bid),
        ask=float(ask),
        delta_bid=float(max(mid - bid, min_delta)),
        delta_ask=float(max(ask - mid, min_delta)),
    )


class AlphaASReplayEnv(gym.Env if _GYMNASIUM_AVAILABLE else object):
    """Single-trajectory replay environment for Alpha-AS-1.

    The action is discrete and selects ``(gamma, skew)``.  The environment then
    holds that pair for a 5-second trading window while A-S formulas generate
    the actual bid/ask quotes at every snapshot.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        midprices: np.ndarray,
        dt_array: np.ndarray,
        *,
        sigma: float,
        A: float,
        kappa: float,
        terminal_time: float | None = None,
        tick_size: float = 0.00001,
        q_max: int = 10,
        decision_interval_sec: float = 5.0,
        seed: int | None = None,
        action_grid: tuple[AlphaASAction, ...] | None = None,
        reward_loss_multiplier: float = 2.0,
    ):
        if not _GYMNASIUM_AVAILABLE:
            raise ImportError("gymnasium is required to instantiate AlphaASReplayEnv.")

        self.midprices = np.asarray(midprices, dtype=np.float64)
        self.dt_array = np.asarray(dt_array, dtype=np.float64)
        if self.midprices.ndim != 1 or self.dt_array.ndim != 1:
            raise ValueError("midprices and dt_array must be one-dimensional arrays.")
        if len(self.midprices) != len(self.dt_array):
            raise ValueError("midprices and dt_array must have the same length.")
        if len(self.midprices) < 2:
            raise ValueError("At least two snapshots are required.")

        self.sigma = float(sigma)
        self.A = float(A)
        self.kappa = float(kappa)
        self.terminal_time = float(terminal_time if terminal_time is not None else self.dt_array.sum())
        self.tick_size = float(tick_size)
        self.q_max = int(q_max)
        self.decision_interval_sec = float(decision_interval_sec)
        self.action_grid = action_grid or build_alpha_as_action_grid()
        self.reward_loss_multiplier = float(reward_loss_multiplier)

        self.action_space = spaces.Discrete(len(self.action_grid))
        self.observation_space = spaces.Box(
            low=-np.ones(32, dtype=np.float32),
            high=np.ones(32, dtype=np.float32),
            dtype=np.float32,
        )
        self.rng = np.random.default_rng(seed)

        self.step_index = 0
        self.cash = 0.0
        self.inventory = 0.0
        self.cumulative_reward = 0.0
        self.inventory_history = deque([0.0] * 5, maxlen=5)
        self.reward_history = deque([0.0] * 5, maxlen=5)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.step_index = 0
        self.cash = 0.0
        self.inventory = 0.0
        self.cumulative_reward = 0.0
        self.inventory_history = deque([0.0] * 5, maxlen=5)
        self.reward_history = deque([0.0] * 5, maxlen=5)
        return self._obs(), {}

    def step(self, action: int):
        action_spec = decode_alpha_as_action(int(action), self.action_grid)
        start_index = self.step_index
        start_mid = float(self.midprices[self.step_index])
        start_pnl = self.cash + self.inventory * start_mid
        start_inventory = self.inventory

        elapsed = 0.0
        sum_spread = 0.0
        quote_count = 0
        while self.step_index < len(self.midprices) - 1 and elapsed < self.decision_interval_sec:
            next_index = self.step_index + 1
            dt = float(self.dt_array[next_index])
            current_mid = float(self.midprices[self.step_index])
            quote = compute_alpha_as_quote(
                midprice=current_mid,
                inventory=self.inventory,
                time_elapsed=float(self.dt_array[: self.step_index + 1].sum()),
                gamma=action_spec.gamma,
                skew=action_spec.skew,
                sigma=self.sigma,
                kappa=self.kappa,
                terminal_time=self.terminal_time,
                tick_size=self.tick_size,
            )

            if dt > 0:
                p_arrival = 1.0 - np.exp(-self.A * dt)
                arrived_bid = self.rng.uniform() < p_arrival
                arrived_ask = self.rng.uniform() < p_arrival
                filled_bid = arrived_bid and self.rng.uniform() < np.exp(-self.kappa * quote.delta_bid)
                filled_ask = arrived_ask and self.rng.uniform() < np.exp(-self.kappa * quote.delta_ask)

                if filled_bid and self.inventory < self.q_max:
                    self.inventory += 1.0
                    self.cash -= quote.bid
                if filled_ask and self.inventory > -self.q_max:
                    self.inventory -= 1.0
                    self.cash += quote.ask

            sum_spread += quote.delta_bid + quote.delta_ask
            quote_count += 1
            self.step_index = next_index
            elapsed += max(dt, 0.0)

        current_mid = float(self.midprices[self.step_index])
        end_pnl = self.cash + self.inventory * current_mid
        pnl_change = end_pnl - start_pnl
        speculative_pnl = start_inventory * (current_mid - start_mid)
        spread_component = pnl_change - speculative_pnl
        reward = spread_component if speculative_pnl >= 0 else spread_component + self.reward_loss_multiplier * speculative_pnl

        self.cumulative_reward += reward
        self.inventory_history.append(np.clip(self.inventory / max(self.q_max, 1), -1.0, 1.0))
        self.reward_history.append(np.clip(reward / max(abs(current_mid), self.tick_size), -1.0, 1.0))

        terminated = self.step_index >= len(self.midprices) - 1
        info = {
            "pnl": float(end_pnl),
            "inventory": float(self.inventory),
            "midprice": current_mid,
            "gamma": action_spec.gamma,
            "skew": action_spec.skew,
            "mean_spread": float(sum_spread / max(quote_count, 1)),
            "snapshots": int(self.step_index - start_index),
        }
        return self._obs(), float(reward), terminated, False, info

    def _obs(self) -> np.ndarray:
        idx = self.step_index
        returns = np.diff(self.midprices[max(0, idx - 10) : idx + 1])
        returns_ticks = np.zeros(10, dtype=np.float64)
        if returns.size:
            returns_ticks[-returns.size :] = returns / max(self.tick_size, 1e-12)
        returns_ticks = np.clip(returns_ticks / 10.0, -1.0, 1.0)

        recent = self.midprices[max(0, idx - 20) : idx + 1]
        current = float(self.midprices[idx])
        start = float(self.midprices[0])
        dt_cum = float(self.dt_array[: idx + 1].sum())

        rolling = []
        for window in (3, 5, 10):
            vals = np.diff(self.midprices[max(0, idx - window) : idx + 1])
            rolling.append(float(np.clip(vals.mean() / max(self.tick_size, 1e-12) / 10.0, -1.0, 1.0)) if vals.size else 0.0)
            rolling.append(float(np.clip(vals.std() / max(self.tick_size, 1e-12) / 10.0, -1.0, 1.0)) if vals.size else 0.0)

        market_features = list(returns_ticks)
        market_features.extend(rolling)
        market_features.extend([
            float(np.clip((current - start) / max(start, self.tick_size), -1.0, 1.0)),
            float(np.clip((current - recent.mean()) / max(current, self.tick_size), -1.0, 1.0)),
            float(np.clip(dt_cum / max(self.terminal_time, 1e-12), 0.0, 1.0)),
            float(np.clip(len(recent) / 20.0, 0.0, 1.0)),
        ])
        market_features.extend([0.0] * (22 - len(market_features)))

        features = list(self.inventory_history) + list(self.reward_history) + market_features[:22]
        return np.asarray(features, dtype=np.float32)


@dataclass
class DoubleDQNConfig:
    total_steps: int = 50_000
    replay_buffer_size: int = 10_000
    batch_size: int = 64
    learning_rate: float = 1e-3
    discount: float = 0.95
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 20_000
    learning_starts: int = 1_000
    train_frequency: int = 1
    target_update_frequency: int = 2
    seed: int = 42


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int):
        self.capacity = int(capacity)
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0

    def add(self, obs, action, reward, next_obs, done) -> None:
        self.obs[self.position] = obs
        self.actions[self.position] = int(action)
        self.rewards[self.position] = float(reward)
        self.next_obs[self.position] = next_obs
        self.dones[self.position] = float(done)
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator):
        idx = rng.integers(0, self.size, size=batch_size)
        return (
            self.obs[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_obs[idx],
            self.dones[idx],
        )


def _build_q_network(obs_dim: int = 32, n_actions: int = 20):
    try:
        import torch.nn as nn
    except Exception as exc:  # pragma: no cover
        raise ImportError("PyTorch is required for Alpha-AS Double DQN.") from exc
    return nn.Sequential(
        nn.Linear(obs_dim, 104),
        nn.ReLU(),
        nn.Linear(104, 104),
        nn.ReLU(),
        nn.Linear(104, n_actions),
    )


def train_double_dqn(
    env_factory: Callable[[], AlphaASReplayEnv],
    config: DoubleDQNConfig | None = None,
    *,
    model_path: str | Path | None = None,
):
    """Train a compact Double DQN for Alpha-AS-1."""
    try:
        import torch
        import torch.nn.functional as F
    except Exception as exc:  # pragma: no cover
        raise ImportError("PyTorch is required for Alpha-AS Double DQN.") from exc

    cfg = config or DoubleDQNConfig()
    rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)

    env = env_factory()
    obs_dim = int(env.observation_space.shape[0])
    n_actions = int(env.action_space.n)
    policy = _build_q_network(obs_dim, n_actions)
    target = _build_q_network(obs_dim, n_actions)
    target.load_state_dict(policy.state_dict())
    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.learning_rate)
    buffer = ReplayBuffer(cfg.replay_buffer_size, obs_dim)

    obs, _ = env.reset(seed=cfg.seed)
    losses: list[float] = []
    episode_rewards: list[float] = []
    current_episode_reward = 0.0
    train_updates = 0

    for step in range(cfg.total_steps):
        frac = min(step / max(cfg.epsilon_decay_steps, 1), 1.0)
        epsilon = cfg.epsilon_start + frac * (cfg.epsilon_end - cfg.epsilon_start)
        if rng.uniform() < epsilon:
            action = int(env.action_space.sample())
        else:
            with torch.no_grad():
                q_values = policy(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))
            action = int(torch.argmax(q_values, dim=1).item())

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = bool(terminated or truncated)
        buffer.add(obs, action, reward, next_obs, done)
        current_episode_reward += float(reward)
        obs = next_obs

        if done:
            episode_rewards.append(current_episode_reward)
            current_episode_reward = 0.0
            env = env_factory()
            obs, _ = env.reset(seed=cfg.seed + len(episode_rewards))

        if (
            buffer.size >= max(cfg.learning_starts, cfg.batch_size)
            and step % cfg.train_frequency == 0
        ):
            batch = buffer.sample(cfg.batch_size, rng)
            obs_b, action_b, reward_b, next_obs_b, done_b = [
                torch.as_tensor(x) for x in batch
            ]
            action_b = action_b.long()
            q = policy(obs_b).gather(1, action_b.reshape(-1, 1)).squeeze(1)
            with torch.no_grad():
                next_actions = policy(next_obs_b).argmax(dim=1)
                next_q = target(next_obs_b).gather(1, next_actions.reshape(-1, 1)).squeeze(1)
                y = reward_b + cfg.discount * (1.0 - done_b) * next_q
            loss = F.mse_loss(q, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

            train_updates += 1
            if train_updates % cfg.target_update_frequency == 0:
                target.load_state_dict(policy.state_dict())

    if model_path is not None:
        torch.save(policy.state_dict(), str(model_path))

    return policy, {
        "losses": losses,
        "episode_rewards": episode_rewards,
        "train_updates": train_updates,
    }


def evaluate_alpha_as_policy(
    policy,
    env_factory: Callable[[int], AlphaASReplayEnv],
    *,
    rollouts: int = 20,
    seed: int = 42,
) -> dict[str, float]:
    """Evaluate a trained Alpha-AS policy and return Falces-style metrics."""
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        raise ImportError("PyTorch is required for Alpha-AS policy evaluation.") from exc

    rollout_rows = []
    for rollout in range(rollouts):
        env = env_factory(seed + rollout)
        obs, _ = env.reset(seed=seed + rollout)
        pnl_path = [0.0]
        inventory_path = [0.0]
        spread_path = []
        done = False
        while not done:
            with torch.no_grad():
                q_values = policy(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))
            action = int(torch.argmax(q_values, dim=1).item())
            obs, _, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            pnl_path.append(float(info["pnl"]))
            inventory_path.append(float(info["inventory"]))
            spread_path.append(float(info["mean_spread"]))

        pnl_arr = np.asarray(pnl_path, dtype=float)
        returns = np.diff(pnl_arr)
        mean_r = returns.mean() if returns.size else 0.0
        std_r = returns.std() if returns.size else 0.0
        neg = returns[returns < 0]
        sortino_std = neg.std() if neg.size >= 2 else 0.0
        running_max = np.maximum.accumulate(pnl_arr)
        max_dd = float(np.max(running_max - pnl_arr)) if pnl_arr.size else 0.0
        mean_abs_q = float(np.mean(np.abs(inventory_path)))
        rollout_rows.append({
            "Sharpe": float(mean_r / std_r) if std_r > 1e-15 else 0.0,
            "Sortino": float(mean_r / sortino_std) if sortino_std > 1e-15 else 0.0,
            "Max DD": max_dd,
            "P&L-to-MAP": float(pnl_arr[-1] / mean_abs_q) if mean_abs_q > 1e-15 else 0.0,
            "Final PnL": float(pnl_arr[-1]),
            "Mean |q|": mean_abs_q,
            "Terminal q": float(inventory_path[-1]),
            "Mean spread": float(np.mean(spread_path)) if spread_path else 0.0,
            "Rollouts": 1.0,
        })

    return pd.DataFrame(rollout_rows).mean(numeric_only=True).to_dict()
