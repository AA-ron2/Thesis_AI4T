"""
Trajectory generation helper.

Pre-allocates 3-D arrays and fills them in a while-loop — exact replica
of ``mbt_gym.gym.helpers.generate_trajectory`` (Jerome et al., 2023).

Return shapes
-------------
observations : (num_trajectories, obs_dim, n_steps + 1)
actions      : (num_trajectories, action_dim, n_steps)
rewards      : (num_trajectories, 1, n_steps)

The 3-D layout ``(N, feature, time)`` is the mbt-gym convention.
Squeezing axis 0 recovers the single-trajectory 2-D view.
"""

from __future__ import annotations

import numpy as np

from mbt_gym.gym.trading_environment import TradingEnvironment
from mbt_gym.agents.agent import Agent


def generate_trajectory(
    env: TradingEnvironment,
    agent: Agent,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Roll out a full episode, collecting observations, actions, rewards.

    Follows the exact logic of ``mbt_gym/gym/helpers/generate_trajectory.py``:
    pre-allocate arrays using ``env.n_steps``, then fill column-by-column.

    Parameters
    ----------
    env : TradingEnvironment
    agent : Agent
    seed : int | None

    Returns
    -------
    observations : np.ndarray, shape (num_trajectories, obs_dim, n_steps + 1)
    actions      : np.ndarray, shape (num_trajectories, action_dim, n_steps)
    rewards      : np.ndarray, shape (num_trajectories, 1, n_steps)
    """
    if seed is not None:
        env.seed(seed)

    obs_space_dim = env.observation_space.shape[0]
    action_space_dim = env.action_space.shape[0]

    observations = np.zeros((env.num_trajectories, obs_space_dim, env.n_steps + 1))
    actions = np.zeros((env.num_trajectories, action_space_dim, env.n_steps))
    rewards = np.zeros((env.num_trajectories, 1, env.n_steps))

    obs, _ = env.reset()
    observations[:, :, 0] = obs

    count = 0
    while True:
        action = agent.get_action(obs)

        # Ensure action is (num_trajectories, action_dim)
        if action.ndim == 1:
            action = np.repeat(action.reshape(1, -1), env.num_trajectories, axis=0)

        obs, reward, done, truncated, info = env.step(action)

        actions[:, :, count] = action
        observations[:, :, count + 1] = obs
        rewards[:, :, count] = reward.reshape(-1, 1)

        if (env.num_trajectories > 1 and done[0]) or (env.num_trajectories == 1 and done[0]):
            break
        count += 1

    return observations, actions, rewards
