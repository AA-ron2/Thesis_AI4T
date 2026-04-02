"""
A-S parameter calibration via Optuna.

Tunes the risk-aversion parameter γ to maximise Sharpe ratio on
the training data.  κ and A are estimated from data (not tuned).

Usage::

    from mbt_gym.gym.calibration import tune_gamma
    best_gamma, study = tune_gamma(
        midprices=S, dt_array=dt_sec,
        sigma=sigma, kappa=35000, A=0.8,
        n_trials=100,
    )

Reference: Bergstra & Bengio (2012), Akiba et al. (2019, Optuna).
"""

from __future__ import annotations

import numpy as np

from mbt_gym.gym.helpers.fast_rollout import fast_simulate


def tune_gamma(
    midprices: np.ndarray,
    dt_array: np.ndarray,
    sigma: float,
    kappa: float,
    A: float,
    tick_size: float = 0.00001,
    Q_MAX: int = 50,
    gamma_range: tuple[float, float] = (0.001, 1.0),
    n_trials: int = 100,
    num_trajectories: int = 50,
    seed: int = 42,
    metric: str = "sharpe",
    verbose: bool = True,
):
    """
    Tune γ (risk aversion) via Optuna TPE to maximise Sharpe.

    Parameters
    ----------
    midprices, dt_array : np.ndarray
        Training day data.
    sigma, kappa, A : float
        Estimated model parameters (fixed during tuning).
    gamma_range : tuple
        Search range for γ (log-uniform).
    n_trials : int
        Number of Optuna trials.
    num_trajectories : int
        N for averaging (N=50 is a good speed/accuracy tradeoff).
    metric : str
        'sharpe' or 'sortino' — objective to maximise.

    Returns
    -------
    best_gamma : float
    study : optuna.Study
    """
    try:
        import optuna
    except ImportError:
        raise ImportError("optuna required. pip install optuna")

    T = float(dt_array.sum())

    def objective(trial):
        gamma = trial.suggest_float("gamma", gamma_range[0], gamma_range[1], log=True)
        stats = fast_simulate(
            midprices=midprices, dt_array=dt_array,
            gamma=gamma, sigma=sigma, kappa=kappa, A=A,
            terminal_time=T, tick_size=tick_size, Q_MAX=Q_MAX,
            num_trajectories=num_trajectories, seed=seed,
            use_linear_approximation=False,
        )
        return float(stats[metric].mean())

    if not verbose:
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    best_gamma = study.best_params["gamma"]
    if verbose:
        print(f"\nBest γ = {best_gamma:.6f}")
        print(f"Best {metric} = {study.best_value:.6f}")

    return best_gamma, study
