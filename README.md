# THESIS_AI4T

Beyond Reward Shaping: CVaR-Constrained Reinforcement Learning for Avellaneda-Stoikov Market Making

This repository contains the code, notebooks, and experiment utilities for my AI4Trading thesis project on risk-aware reinforcement learning for market making. The project studies whether we can improve on classical Avellaneda-Stoikov and unconstrained PPO agents without accepting the large drawdowns often seen in RL trading agents.

The central idea is simple:

> We do not penalise drawdowns, we constrain them.

Instead of adding a fixed drawdown penalty to the reward, this thesis formulates market making as a constrained Markov decision process (CMDP) and trains a PPO agent with a CVaR-based drawdown constraint enforced through Lagrangian relaxation.

## Research Focus

The thesis compares four families of agents:

- Avellaneda-Stoikov analytical market-making baseline
- Unconstrained PPO
- Drawdown-penalised PPO
- CVaR-constrained PPO-Lagrangian

The main optimisation problem is:

```text
max    E[sum_t gamma^t r_t]
such that    CVaR_alpha(MaxDrawdown) <= d
```

where the constraint targets tail drawdown risk rather than average drawdown.

## Repository Structure

```text
THESIS_AI4T/
|- procs/                       # Reusable thesis code package
|  |- agents/                   # A-S baseline agent and SB3 agent wrapper
|  |- gym/                      # Environments, calibration, metrics, config, CVaR logic
|  |- rewards/                  # Reward functions used in training/evaluation
|  `- stochastic_processes/     # Midprice, arrival, and fill models
|- notebooks/
|  |- nb1_bm_validation.ipynb   # Brownian motion validation stage
|  |- nb2_dd_penalised.ipynb    # Drawdown-penalised PPO on replay data
|  |- nb3_cvar_constrained.ipynb# CVaR-constrained PPO on replay data
|  `- nb_as_baseline_30day.ipynb# Multi-day A-S baseline analysis
|- datasets/                    # Local market data (ignored by git)
|- models/                      # Saved SB3 models / VecNormalize stats
|- results/                     # Tables, figures, exported outputs
`- tests/                       # Lightweight regression checks
```

Although the architecture follows the ideas of `mbt-gym`, the working code in this repository lives under [`procs/`](procs/).

## Data

The replay experiments use Binance perpetual futures order book data:

- Asset: `DOGEUSDT`
- Source: Tardis L2 order book snapshots
- Depth: 25 levels per side
- Period used in the thesis: January 2025
- Expected filename pattern:
  `binance_book_snapshot_25_YYYY-MM-DD_DOGEUSDT.csv`

The loader in [`procs/gym/data_loader.py`](procs/gym/data_loader.py) expects CSVs in the local [`datasets/`](datasets/) folder and computes the midprice as:

```text
midprice = (asks[0].price + bids[0].price) / 2
```

## Main Components

- [`procs/gym/cvar_lagrangian.py`](procs/gym/cvar_lagrangian.py): core thesis contribution, including drawdown-cost tracking, CVaR estimation, and the PPO-Lagrangian callback
- [`procs/gym/notebook_support.py`](procs/gym/notebook_support.py): helper builders for replay environments, evaluation, and sensitivity analysis
- [`procs/gym/experiment_config.py`](procs/gym/experiment_config.py): repo-relative experiment configuration for Brownian-motion and replay experiments
- [`procs/gym/helpers/fast_rollout.py`](procs/gym/helpers/fast_rollout.py): fast NumPy-only rollout path for analytical baselines
- [`procs/rewards/reward_funcs.py`](procs/rewards/reward_funcs.py): `PnLReward`, `CjMmCriterion`, and `CjMmDrawdownPenalty`

## Notebook Workflow

The notebooks are the main entry point for reproducing the thesis experiments:

1. `nb1_bm_validation.ipynb`
   Validates the environment and compares PPO against the Avellaneda-Stoikov baseline in a Brownian motion setting.

2. `nb2_dd_penalised.ipynb`
   Runs the replay-data experiment with a fixed drawdown penalty baseline.

3. `nb3_cvar_constrained.ipynb`
   Trains the CVaR-constrained PPO agent with threshold calibration and Lagrangian updates.

4. `nb_as_baseline_30day.ipynb`
   Runs broader baseline analysis for the analytical A-S strategy across multiple replay days.

## Setup

This repo currently does not include a pinned `requirements.txt` or environment file, so the setup below is the practical baseline inferred from the code imports and notebook workflow.

### 1. Create an environment

```powershell
conda create -n mysimenv python=3.11
conda activate mysimenv
```

### 2. Install dependencies

```powershell
pip install numpy pandas matplotlib seaborn pytest jupyterlab gymnasium optuna "stable-baselines3[extra]"
```

`stable-baselines3[extra]` pulls in the RL stack and TensorBoard support used during training.

### 3. Add market data

Place your Tardis CSV files in:

```text
datasets/
```

For example:

```text
datasets/binance_book_snapshot_25_2025-01-01_DOGEUSDT.csv
```

### 4. Launch notebooks

Start Jupyter from the repository root:

```powershell
jupyter lab
```

The repo includes path helpers so notebooks can reliably find the project root when run from either the root directory or `notebooks/`.

## Quick Smoke Test

Once data is available, you can verify that the project wiring works with:

```powershell
pytest tests/test_experiment_support.py
```

You can also test data loading directly:

```python
from procs.gym.experiment_config import ReplayExperimentConfig
from procs.gym.data_loader import load_single_day

cfg = ReplayExperimentConfig()
S, dt, index = load_single_day(str(cfg.data_path()))

print(len(S), "snapshots loaded")
print("Total horizon (sec):", float(dt.sum()))
```

## Metrics Used

The evaluation utilities report the same family of metrics used throughout the thesis:

- Sharpe ratio
- Sortino ratio
- Maximum drawdown
- P&L-to-MAP
- Final PnL
- Mean absolute inventory

See [`procs/gym/helpers/generate_trajectory_stats.py`](procs/gym/helpers/generate_trajectory_stats.py) and [`procs/gym/metrics.py`](procs/gym/metrics.py).

## Key Implementation Notes

- The reusable Python package in this repo is `procs`, not `mbt_gym`.
- Replay experiments are built around `DOGEUSDT` market data and repo-relative paths from `ReplayExperimentConfig`.
- The project is notebook-driven: models, plots, and result tables are generated from the notebooks and saved into local artifact folders such as `models/` and `results/`.
- Dataset files are intentionally not versioned in Git.

## References

This project is built around the intersection of:

- Avellaneda and Stoikov (2008) for analytical market making
- Falces Marin et al. (2022) for RL market-making evaluation and the drawdown motivation
- Altman (1999), Rockafellar and Uryasev (2000), Chow et al. (2015), and Ray et al. (2019) for CMDPs, CVaR, and Lagrangian safe RL
- Jerome et al. (2023) for the `mbt-gym` architectural inspiration

## Status

This repository is an active thesis workspace. The notebooks contain the experimental narrative, while `procs/` holds the reusable environment, baseline, calibration, and CVaR-constrained training code.
