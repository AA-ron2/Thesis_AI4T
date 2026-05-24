import numpy as np
import pytest

import importlib.util
import sys
from pathlib import Path


_MODULE_PATH = Path(__file__).resolve().parents[1] / "procs" / "gym" / "alpha_as.py"
_SPEC = importlib.util.spec_from_file_location("alpha_as_for_tests", _MODULE_PATH)
alpha_as = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
sys.modules[_SPEC.name] = alpha_as
_SPEC.loader.exec_module(alpha_as)


def test_alpha_as_action_grid_has_expected_20_pairs():
    grid = alpha_as.build_alpha_as_action_grid()

    assert len(grid) == 20
    assert alpha_as.decode_alpha_as_action(0, grid).gamma == 0.01
    assert alpha_as.decode_alpha_as_action(0, grid).skew == -0.1
    assert alpha_as.decode_alpha_as_action(19, grid).gamma == 0.9
    assert alpha_as.decode_alpha_as_action(19, grid).skew == 0.1
    assert sorted({action.gamma for action in grid}) == [0.01, 0.1, 0.2, 0.9]
    assert sorted({action.skew for action in grid}) == [-0.1, -0.05, 0.0, 0.05, 0.1]


def test_alpha_as_quote_depths_are_nonnegative_and_not_crossed():
    quote = alpha_as.compute_alpha_as_quote(
        midprice=100.0,
        inventory=3,
        time_elapsed=5.0,
        gamma=0.1,
        skew=0.1,
        sigma=0.02,
        kappa=1.5,
        terminal_time=100.0,
        tick_size=0.01,
    )

    assert quote.delta_bid > 0
    assert quote.delta_ask > 0
    assert quote.bid < quote.ask
    assert quote.bid <= 100.0
    assert quote.ask >= 100.0


def _small_env(seed=42):
    if not alpha_as._GYMNASIUM_AVAILABLE:
        pytest.skip("gymnasium is required for AlphaASReplayEnv")
    midprices = np.array([100.00, 100.01, 100.00, 100.02, 100.01, 100.03])
    dt = np.array([0.0, 1.0, 1.2, 1.0, 0.8, 1.1])
    return alpha_as.AlphaASReplayEnv(
        midprices,
        dt,
        sigma=0.02,
        A=0.8,
        kappa=1.5,
        terminal_time=float(dt.sum()),
        tick_size=0.01,
        q_max=10,
        decision_interval_sec=2.0,
        seed=seed,
    )


def test_alpha_as_env_step_and_termination_smoke():
    env = _small_env()
    obs, _ = env.reset(seed=123)

    assert obs.shape == (32,)
    next_obs, reward, terminated, truncated, info = env.step(0)
    assert next_obs.shape == (32,)
    assert isinstance(reward, float)
    assert not truncated
    assert {"pnl", "inventory", "gamma", "skew", "mean_spread"} <= set(info)

    while not terminated:
        _, _, terminated, truncated, _ = env.step(0)
    assert terminated
    assert not truncated


def test_double_dqn_tiny_smoke():
    pytest.importorskip("torch")

    cfg = alpha_as.DoubleDQNConfig(
        total_steps=25,
        replay_buffer_size=100,
        batch_size=4,
        learning_starts=4,
        train_frequency=1,
        target_update_frequency=2,
        seed=7,
    )
    policy, history = alpha_as.train_double_dqn(lambda: _small_env(seed=7), cfg)

    assert policy is not None
    assert history["train_updates"] > 0
