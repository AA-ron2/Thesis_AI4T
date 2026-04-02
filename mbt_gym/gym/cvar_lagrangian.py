"""
CVaR-Constrained PPO via Lagrangian relaxation.

Implements the core thesis contribution: "We do not penalise drawdowns
— we constrain them."

The standard PPO objective is:
    max_θ  J(θ) = E[Σ γ^t r_t]

The CVaR-CMDP adds a constraint:
    CVaR_α( max_drawdown ) ≤ d_max

The Lagrangian relaxation converts this to:
    max_θ min_λ  J(θ) − λ · ( CVaR_α(DD) − d_max )

Bug-fix history
---------------
Bug 1 (Stage 1 threshold): median of test-time DD set threshold infeasibly low.
    Fix: calibrate_cvar_threshold() uses CVaR_α of unconstrained policy rollouts.

Bug 2 (Stage 2 threshold): calibrated from A-S baseline, not unconstrained PPO.
    Fix: same calibrate_cvar_threshold() called independently per stage.

Bug 3 (λ update unscaled): raw violation → λ saturated in ~18 iters.
    Fix: normalised update λ += η·(CVaR−d)/max(d,ε); lr=0.01.

Bug 4 (λ_max too small): cap of 50 saturated immediately.
    Fix: λ_max=500.

Bug 5 (DOGE episodes never terminate within rollout): episode_max_drawdowns
    always empty for 714k-step episodes with n_steps=2048 rollouts.
    Fix: primary CVaR signal uses per-window max DD from step_costs.

Bug 6 (warm-start mismatch): CVaR PPO starts from random weights, but
    calibration was done on the converged unconstrained policy. Early rollouts
    have CVaR ≫ d, constraint infeasible from iteration 0, λ grows without
    the policy being able to respond.
    Fix: train_cvar_ppo accepts model_init to warm-start from unconstrained PPO.

Bug 7 (calibration/training window mismatch for DOGE): calibrate_cvar_threshold
    runs full 714k-step episodes; training rollouts are 2048-step windows.
    MaxDD_window ≈ MaxDD_episode × sqrt(2048/714000) ≈ 5% of calibration value.
    Threshold is always above training CVaR → λ never activates.
    Fix: calibrate_cvar_threshold_windowed() samples windows of size n_steps.

Bug 8 (terminal-state reset inflates training CVaR by 7-8x, Stage 1):
    StableBaselinesTradingEnvironment.step_wait() calls env.reset() inside
    step_wait when dones.min()==True, BEFORE returning. By the time
    DrawdownCostWrapper reads model_dynamics.state, it sees the post-reset
    initial state: cash=0, inventory=0, price=S0. This gives PnL=0, so:
        drawdown = peak_pnl - 0 = peak_pnl ≈ 50-70 price units
    This artificial number is appended to step_costs every rollout (at the
    terminal step), inflating the training CVaR from the true ~9.4 to ~68.
    Fix: on terminal steps, do NOT read model_dynamics.state. Instead use
    _episode_max_dd (computed correctly from non-terminal steps) as the
    cost for the terminal step, then reset tracking for the new episode.

References:
    • Altman (1999), Constrained Markov Decision Processes
    • Achiam et al. (2017), Constrained Policy Optimization
    • Ray et al. (2019), Benchmarking Safe Exploration in Deep RL
    • Rockafellar & Uryasev (2000), Optimization of CVaR
    • Chow et al. (2015), Risk-Constrained RL with Percentile Risk Criteria
    • Stooke et al. (2020), Responsive Safety in RL by PID Lagrangian Methods
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import VecEnv
    _SB3_AVAILABLE = True
except ImportError:
    _SB3_AVAILABLE = False
    BaseCallback = object

    class VecEnv:
        def __init__(self, *a, **kw): ...


# ═══════════════════════════════════════════════════════════════
# 1. DrawdownCostWrapper
# ═══════════════════════════════════════════════════════════════

class DrawdownCostWrapper(VecEnv):
    """
    VecEnv wrapper that tracks per-step drawdown cost.

    At each step:  c_t = max(0, peak_PnL_so_far − current_PnL)
    """

    def __init__(self, sb3_env):
        self.env = sb3_env
        self.N = sb3_env.num_envs
        super().__init__(
            num_envs=self.N,
            observation_space=sb3_env.observation_space,
            action_space=sb3_env.action_space,
        )
        self._peak_pnl = np.zeros(self.N)
        self._episode_max_dd = np.zeros(self.N)
        self.step_costs: list[np.ndarray] = []
        self.episode_max_drawdowns: list[np.ndarray] = []

    def reset(self) -> np.ndarray:
        obs = self.env.reset()
        self._peak_pnl[:] = 0.0
        self._episode_max_dd[:] = 0.0
        self.step_costs = []
        return obs

    def step_async(self, actions: np.ndarray) -> None:
        self.env.step_async(actions)

    def step_wait(self):
        obs, rewards, dones, infos = self.env.step_wait()

        # ── Bug 8 fix: terminal-state reset ──────────────────────────────────
        #
        # StableBaselinesTradingEnvironment.step_wait() auto-resets the
        # environment when dones.min()==True — it calls env.reset() before
        # returning. By the time we read model_dynamics.state here, the
        # environment is already in its *initial* state for the new episode:
        # cash=0, inventory=0, price=S0.
        #
        # Computing PnL from this reset state gives PnL=0, so:
        #     drawdown = peak_pnl - 0 = peak_pnl ≈ 50–70 price units
        # This artificial "drawdown" is appended to step_costs and dominates
        # the CVaR calculation every single rollout, inflating the training
        # CVaR by 7–8× vs the calibrated threshold.
        #
        # Fix: on terminal steps (dones.min()==True), do NOT read
        # model_dynamics.state. Instead, record _episode_max_dd (computed
        # correctly from all non-terminal steps) as the cost for this step,
        # then reset tracking state for the new episode.
        #
        # The step_costs list always has exactly one entry per step, so
        # cost_array shape in _on_rollout_end remains (n_steps, N). ✓
        #
        # Reference: this is the same issue as described in SB3's documentation
        # on handling terminal observations in vectorised environments —
        # model state must be read BEFORE the auto-reset, not after.

        if dones.min():
            # Finalise episode: use the running max DD (valid, pre-reset)
            # as the cost for this terminal step. Reset for next episode.
            self.episode_max_drawdowns.append(self._episode_max_dd.copy())
            self.step_costs.append(self._episode_max_dd.copy())  # NOT post-reset PnL=0
            self._peak_pnl[:] = 0.0
            self._episode_max_dd[:] = 0.0
        else:
            # Mid-episode: model_dynamics.state is valid (no reset has occurred)
            raw_state = self.env.env.model_dynamics.state
            if raw_state is not None:
                pnl = raw_state[:, 0] + raw_state[:, 1] * raw_state[:, 3]
            else:
                pnl = np.zeros(self.N)
            self._peak_pnl = np.maximum(self._peak_pnl, pnl)
            drawdown = self._peak_pnl - pnl
            self._episode_max_dd = np.maximum(self._episode_max_dd, drawdown)
            self.step_costs.append(drawdown.copy())

        return obs, rewards, dones, infos

    def close(self) -> None:              self.env.close()
    def seed(self, seed=None):            return self.env.seed(seed)
    def get_attr(self, a, indices=None):  return self.env.get_attr(a, indices)
    def set_attr(self, a, v, indices=None): self.env.set_attr(a, v, indices)
    def env_method(self, m, *a, indices=None, **kw): return self.env.env_method(m, *a, indices=indices, **kw)
    def env_is_wrapped(self, w, indices=None): return self.env.env_is_wrapped(w, indices)
    def get_images(self):                 return self.env.get_images()


# ═══════════════════════════════════════════════════════════════
# 2. CVaRLagrangianCallback
# ═══════════════════════════════════════════════════════════════

class CVaRLagrangianCallback(BaseCallback):
    """
    SB3 callback implementing the Lagrangian dual ascent update for the
    CVaR constraint on maximum drawdown.

    After each rollout:
      1. Collect per-window max DD from step_costs (Bug 5 fix)
      2. Estimate CVaR_α
      3. Normalised dual ascent: λ += η·(CVaR−d)/max(d,ε)  (Bug 3 fix)
      4. Penalise rollout buffer: r' = r − λ·c_t

    Parameters
    ----------
    dd_threshold : float
        Must be obtained via calibrate_cvar_threshold_windowed() for long
        episodes (DOGE), or calibrate_cvar_threshold() for short episodes
        (BM). Both must use the *warm-started* policy as the reference,
        which is now automatic since train_cvar_ppo warm-starts.
    lr_lambda : float
        Dual LR on *normalised* violation. Default 0.01.
    lambda_max : float
        Default 500. With lr=0.01, saturation requires 50k rollouts of
        100% violation — never happens once the policy adapts.
    """

    def __init__(
        self,
        cost_wrapper: DrawdownCostWrapper,
        cvar_alpha: float = 0.2,
        dd_threshold: float = 0.05,
        lr_lambda: float = 0.01,
        lambda_init: float = 0.0,
        lambda_max: float = 500.0,
        verbose: int = 1,
    ):
        super().__init__(verbose=verbose)
        self.cost_wrapper = cost_wrapper
        self.cvar_alpha = cvar_alpha
        self.dd_threshold = dd_threshold
        self.lr_lambda = lr_lambda
        self.lambda_ = lambda_init
        self.lambda_max = lambda_max
        self.cvar_history: list[float] = []
        self.lambda_history: list[float] = []

    def _on_rollout_end(self) -> None:
        # ── 1. Per-window max DD (Bug 5 fix) ─────────────────
        if len(self.cost_wrapper.step_costs) > 0:
            cost_matrix = np.array(self.cost_wrapper.step_costs)  # (T, N)
            all_dd = cost_matrix.max(axis=0)                      # (N,)
        elif len(self.cost_wrapper.episode_max_drawdowns) > 0:
            all_dd = np.concatenate(self.cost_wrapper.episode_max_drawdowns)
        else:
            all_dd = self.cost_wrapper._episode_max_dd.copy()

        if len(all_dd) == 0:
            return

        # ── 2. CVaR_α ────────────────────────────────────────
        sorted_dd = np.sort(all_dd)
        cutoff_idx = max(1, int(np.ceil(len(sorted_dd) * self.cvar_alpha)))
        cvar = sorted_dd[-cutoff_idx:].mean()

        # ── 3. Normalised dual ascent (Bug 3 fix) ─────────────
        violation_norm = (cvar - self.dd_threshold) / max(self.dd_threshold, 1e-8)
        self.lambda_ = float(np.clip(
            self.lambda_ + self.lr_lambda * violation_norm,
            0.0, self.lambda_max,
        ))

        self.cvar_history.append(float(cvar))
        self.lambda_history.append(self.lambda_)

        if self.verbose >= 1:
            print(f"  CVaR={cvar:.6f}  d={self.dd_threshold:.6f}"
                  f"  viol={violation_norm:+.4f}  λ={self.lambda_:.4f}")

        # ── 4. Penalise rewards ───────────────────────────────
        buffer = self.model.rollout_buffer
        n_steps = buffer.rewards.shape[0]
        costs = self.cost_wrapper.step_costs
        if len(costs) >= n_steps:
            cost_array = np.array(costs[-n_steps:])
            buffer.rewards -= self.lambda_ * cost_array.reshape(buffer.rewards.shape)

        self.cost_wrapper.step_costs = []
        self.cost_wrapper.episode_max_drawdowns = []

    def _on_step(self) -> bool:
        return True


# ═══════════════════════════════════════════════════════════════
# 3. Threshold calibration — full-episode version  (BM)
# ═══════════════════════════════════════════════════════════════

def calibrate_cvar_threshold(
    env,
    agent,
    cvar_alpha: float = 0.2,
    n_episodes: int = 1,
    tighten: float = 0.2,
    verbose: bool = True,
) -> tuple[float, float]:
    """
    Calibrate CVaR threshold from the unconstrained policy's *full-episode*
    max-drawdown distribution.

    Use this for BM (n_steps ≥ episode length, so rollout = full episode).
    For DOGE (n_steps ≪ episode length), use calibrate_cvar_threshold_windowed.

    Returns (threshold, cvar_raw) where threshold = cvar_raw × (1 - tighten).
    """
    from mbt_gym.gym.helpers.generate_trajectory_stats import generate_trajectory_stats

    all_dd: list[float] = []
    for _ in range(n_episodes):
        stats = generate_trajectory_stats(env, agent)
        all_dd.extend(stats["max_drawdown"].tolist())

    all_dd_arr = np.array(all_dd)
    sorted_dd = np.sort(all_dd_arr)
    cutoff_idx = max(1, int(np.ceil(len(sorted_dd) * cvar_alpha)))
    cvar_raw = float(sorted_dd[-cutoff_idx:].mean())
    threshold = cvar_raw * (1.0 - tighten)

    if verbose:
        N = len(all_dd_arr)
        print("CVaR threshold calibration (full-episode)")
        print(f"  {n_episodes} episode(s) × {N // max(n_episodes, 1)} trajectories = {N} samples")
        print(f"  CVaR_{cvar_alpha:.0%} = mean of worst {cutoff_idx} = {cvar_raw:.6f}")
        print(f"  Tighten {tighten:.0%}  →  d = {threshold:.6f}")
        print(f"  [mean={all_dd_arr.mean():.6f}  median={np.median(all_dd_arr):.6f}"
              f"  p95={np.percentile(all_dd_arr, 95):.6f}]")

    return threshold, cvar_raw


# ═══════════════════════════════════════════════════════════════
# 4. Threshold calibration — windowed version  (DOGE)  [Bug 7 fix]
# ═══════════════════════════════════════════════════════════════

def calibrate_cvar_threshold_windowed(
    env,
    agent,
    n_steps: int,
    cvar_alpha: float = 0.2,
    n_windows: int = 50,
    tighten: float = 0.2,
    verbose: bool = True,
) -> tuple[float, float]:
    """
    Calibrate CVaR threshold from *n_steps-length window* max drawdowns.

    This fixes Bug 7: calibrate_cvar_threshold runs full episodes (714k steps)
    but training rollouts are only n_steps=2048 steps long. The expected max
    drawdown scales as σ√T, so:

        E[MaxDD_{2048}] ≈ E[MaxDD_{714k}] × sqrt(2048/714000) ≈ 5%

    The threshold must be calibrated on windows of the *same length* as
    training rollouts so the constraint is active from iteration 1.

    Procedure:
        1. Reset env, run agent for n_windows × n_steps steps
        2. Record max drawdown within each n_steps window
        3. Compute CVaR_α across those window max DDs
        4. Return d = CVaR × (1 - tighten)

    Parameters
    ----------
    env : TradingEnvironment (normalise=False, N=1)
    agent : Agent (warm-started unconstrained PPO)
    n_steps : int
        Must match n_steps used in train_cvar_ppo's PPO kwargs.
    cvar_alpha : float
    n_windows : int
        Number of windows to collect. Minimum: ceil(5 / cvar_alpha).
        Default 50 → 50 samples, 10 in CVaR_0.2 tail.
    tighten : float
    verbose : bool

    Returns
    -------
    threshold : float
    cvar_raw : float
    """
    from mbt_gym.gym.index_names import CASH_INDEX, INVENTORY_INDEX, ASSET_PRICE_INDEX

    obs, _ = env.reset()
    N = env.num_trajectories

    window_max_dds: list[float] = []
    peak_pnl = (
        obs[:, CASH_INDEX]
        + obs[:, INVENTORY_INDEX] * obs[:, ASSET_PRICE_INDEX]
    ).copy()
    window_dd = np.zeros(N)

    step = 0
    done = False

    while len(window_max_dds) < n_windows and not done:
        action = agent.get_action(obs)
        if action.ndim == 1:
            action = np.repeat(action.reshape(1, -1), N, axis=0)

        obs, _, term, _, _ = env.step(action)
        pnl = (
            obs[:, CASH_INDEX]
            + obs[:, INVENTORY_INDEX] * obs[:, ASSET_PRICE_INDEX]
        )
        peak_pnl = np.maximum(peak_pnl, pnl)
        window_dd = np.maximum(window_dd, peak_pnl - pnl)

        step += 1

        if step % n_steps == 0:
            # Record this window's max DD per trajectory
            window_max_dds.extend(window_dd.tolist())
            window_dd[:] = 0.0
            # Don't reset peak — drawdown is cumulative from episode start

        if term[0]:
            done = True

    if len(window_max_dds) == 0:
        raise RuntimeError("No windows collected — episode shorter than n_steps?")

    all_dd_arr = np.array(window_max_dds)
    sorted_dd = np.sort(all_dd_arr)
    cutoff_idx = max(1, int(np.ceil(len(sorted_dd) * cvar_alpha)))
    cvar_raw = float(sorted_dd[-cutoff_idx:].mean())
    threshold = cvar_raw * (1.0 - tighten)

    if verbose:
        n_collected = len(all_dd_arr)
        print(f"CVaR threshold calibration (windowed, window={n_steps} steps)")
        print(f"  {n_collected} windows × {N} trajectory = {n_collected * N} samples")
        print(f"  CVaR_{cvar_alpha:.0%} = mean of worst {cutoff_idx} = {cvar_raw:.6f}")
        print(f"  Tighten {tighten:.0%}  →  d = {threshold:.6f}")
        print(f"  [mean={all_dd_arr.mean():.6f}  median={np.median(all_dd_arr):.6f}"
              f"  p95={np.percentile(all_dd_arr, 95):.6f}]")

    return threshold, cvar_raw


# ═══════════════════════════════════════════════════════════════
# 5. train_cvar_ppo — with warm-start support  [Bug 6 fix]
# ═══════════════════════════════════════════════════════════════

def train_cvar_ppo(
    sb3_env,
    total_timesteps: int,
    cvar_alpha: float = 0.2,
    dd_threshold: float = 0.05,
    lr_lambda: float = 0.01,
    lambda_init: float = 0.0,
    lambda_max: float = 500.0,
    ppo_kwargs: dict | None = None,
    model_init=None,               # Bug 6 fix: warm-start from unconstrained PPO
    verbose: int = 1,
):
    """
    Train a CVaR-constrained PPO agent.

    Parameters
    ----------
    model_init : PPO model or None
        If provided, the CVaR PPO is warm-started from these weights via
        set_parameters(). This is critical for correct Lagrangian behaviour:

        Without warm-start: CVaR PPO starts from random weights. Early
        rollouts have CVaR >> d (infeasible region). λ grows uncontrollably
        before the policy can adapt, driving it to a degenerate "don't trade"
        solution (Bug 6).

        With warm-start: early rollouts have CVaR ≈ calibration CVaR (since
        the policy is the same as what was calibrated). The constraint is
        active-but-feasible from iteration 1, and λ guides the policy
        incrementally toward lower drawdown. (Achiam et al. 2017, §5.)

    dd_threshold : float
        For BM: use calibrate_cvar_threshold(unconstrained_agent, ...).
        For DOGE: use calibrate_cvar_threshold_windowed(unconstrained_agent,
                  n_steps=<rollout_n_steps>, ...).
        Both must use the *unconstrained PPO agent* as reference.

    Returns
    -------
    (model, callback, cost_wrapper)
    """
    if not _SB3_AVAILABLE:
        raise ImportError("stable-baselines3 required.")

    cost_wrapper = DrawdownCostWrapper(sb3_env)

    callback = CVaRLagrangianCallback(
        cost_wrapper=cost_wrapper,
        cvar_alpha=cvar_alpha,
        dd_threshold=dd_threshold,
        lr_lambda=lr_lambda,
        lambda_init=lambda_init,
        lambda_max=lambda_max,
        verbose=verbose,
    )

    default_kwargs = dict(
        policy="MlpPolicy",
        verbose=verbose,
        device="cpu",
        n_steps=200,
        batch_size=50_000,
        n_epochs=10,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
    )
    if ppo_kwargs:
        default_kwargs.update(ppo_kwargs)

    policy = default_kwargs.pop("policy")
    model = PPO(policy, cost_wrapper, **default_kwargs)

    # Bug 6 fix: warm-start from unconstrained policy
    if model_init is not None:
        model.set_parameters(model_init.get_parameters())
        if verbose:
            print("CVaR PPO warm-started from unconstrained policy weights.")

    model.learn(total_timesteps=total_timesteps, callback=callback)

    return model, callback, cost_wrapper
