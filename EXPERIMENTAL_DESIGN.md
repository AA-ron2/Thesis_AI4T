# Experimental Design and Notebook Structure
## Beyond Reward Shaping: CVaR-Constrained RL for Avellaneda-Stoikov Market Making

---

## 1. Overview

This document describes the revised experimental design for the four-agent comparison
(B0–B3) used in the thesis. It supersedes the earlier `MODEL_COMPARISON.md` where
relevant, and provides detailed notebook-level structure for each agent.

The central academic contribution is the comparison of **reward shaping** (B2) against
**Lagrangian constraint enforcement** (B3) for drawdown control in a PPO market-making
agent. The design is structured as a direct extension of Falces Marín et al. (2022), which
identified excessive drawdown in RL market-making agents as an open problem.

---

## 2. Dataset and Train/Test Split

### Data
- **Asset:** DOGEUSDT Binance perpetual futures
- **Source:** Tardis L2 order book snapshots, 25 levels per side
- **Resolution:** ~100ms (variable)
- **Period:** January 2025 (29 available trading days)
- **Filename pattern:** `binance_book_snapshot_25_YYYY-MM-DD_DOGEUSDT.csv`

### Split
| Set | Days | Dates | Purpose |
|-----|------|-------|---------|
| Training | 6 | 2025-01-01 to 2025-01-06 | All agents: B0 γ cross-validation; B1/B2/B3 training |
| Test | 23 | 2025-01-07 to 2025-01-29 | All agent evaluation (held-out, never touched during training) |

**Rationale:** The chronological split preserves temporal ordering, which is required for
financial time series — shuffling would introduce look-ahead bias. All four agents use
the same training/test boundary: calibrated or trained on days 1–6, evaluated on days 7–29.

### Regime characteristics
| Set | Mean σ | Max σ | Notes |
|-----|--------|-------|-------|
| Training | ~0.00112 | ~0.00149 | Low-to-moderate volatility regime |
| Test | ~0.00180 | ~0.00365 | Includes Jan 19–21 high-volatility episode |

**Important caveat:** The training set contains no high-volatility days. The test set
includes a significant volatility spike (Jan 19–21), constituting a partial distributional
shift. This is acknowledged as a limitation but also serves as a stress test of
out-of-sample generalisation.

---

## 3. Agent Descriptions

### B0 — Avellaneda-Stoikov Analytical Baseline

**Type:** Analytical (no learning)

**Quoting formulas (Avellaneda & Stoikov 2008):**
```
r(t)   = S(t) − q · γ · σ² · τ          (reservation price)
δ*(t)  = γ · σ² · τ + (2/γ) · ln(1 + γ/κ)  (optimal spread)
δ_bid  = q · γ · σ² · τ + δ*/2
δ_ask  = −q · γ · σ² · τ + δ*/2
```

**Parameter estimation:**
- **σ, κ, A**: Estimated per test day from that day's price data. These are market
  parameters that genuinely vary day-to-day; re-estimation each day mimics realistic
  deployment where a trader re-calibrates each morning. This is methodologically sound
  and does not constitute in-sample evaluation.
- **γ**: Fixed constant, cross-validated on training days 1–6 only via Optuna TPE
  (n_trials ≥ 20). The γ that maximises mean Sharpe across the 6 training days is
  selected and applied unchanged to all 23 test days. γ is never tuned on test data.

**Why fixed γ is the right design:**
γ is a preference parameter (risk aversion), not a market property. Tuning it per test
day on that day's own data constitutes in-sample evaluation — it finds the γ that makes
each specific day look best in hindsight and is not deployable in practice. A fixed γ
calibrated on training data represents a realistic deployment scenario and makes B0
directly comparable to Falces Marín et al. (2022), who also use fixed parameters.
This is consistent with the train/test split applied to B1–B3.

**Role in thesis:**
Establishes the out-of-sample analytical baseline for classical A-S. Near-optimal on
Brownian motion by construction (BM is the generative model A-S was derived under).
Expected to degrade on real DOGE data due to non-BM dynamics (jumps, autocorrelation,
regime changes), providing the motivation for the RL agents.

**Snellius note:** B0 calibration (γ cross-validation on training days) runs via
`baseline_snellius.py` with `--max-days 6`. Test evaluation (23 days, parallelised)
uses the same `baseline_sweep.sh` job array with `--array=0-22`.

---

### B1 — Unconstrained PPO

**Type:** RL (Stable-Baselines3 PPO)

**Reward (Cartea-Jaimungal criterion):**
```
r_t = ΔPnL − φ · q² · Δt       (φ = 0.01)
```

**Training:** Days 1–6 (DOGEUSDT market replay environment)

**Architecture:**
```python
PPO("MlpPolicy", env,
    n_steps=2048, batch_size=512, n_epochs=10,
    learning_rate=3e-4, gamma=0.999,
    gae_lambda=0.95, clip_range=0.2, ent_coef=0.01)
```

**Role in thesis:**
Establishes what RL achieves without any risk management. Expected to match or exceed
B0 on PnL (RL adapts to non-BM market dynamics) but with excessive drawdown — the
open problem identified by Falces Marín et al. (2022) that motivates B3.

**Snellius note:** Training is submitted as a single SLURM job via `run.sh` on the
`genoa` partition. Estimated wall time: 2–4 hours on 6 training days. Save model
checkpoint to `models/` and sync back to local via MobaXterm file browser or SCP.

---

### B2 — Drawdown-Penalised PPO

**Type:** RL (reward shaping)

**Reward:**
```
r_t = ΔPnL − φ · q² · Δt − α · drawdown_t       (φ = 0.01, α swept)
```

**α sweep:** [0.1, 1.0, 10.0] — three separate trained models

**Training:** Days 1–6, identical setup to B1 except reward function

**Role in thesis:**
Ablation baseline for B3. The α sweep produces a Pareto frontier of PnL vs drawdown,
demonstrating that reward shaping can reduce drawdown but introduces a fixed tradeoff
with no convergence guarantee (Altman 1999 §3.4). The best α from the sweep is used
as the primary B2 comparison point against B3.

**Known weaknesses of reward shaping (to be argued in thesis):**
1. No single α achieves both B1-level PnL and B3-level drawdown simultaneously
2. Penalty distorts the value function — the agent learns to avoid drawdown-generating
   states not because of the constraint but because they reduce reward signal
3. No formal convergence guarantee to the constrained optimum (unlike CMDP)

**Snellius note:** The α sweep (3 configs) is submitted as a SLURM job array
(`--array=0-2`), one job per α value. Each job trains independently. All three
models and their VecNormalize stats are saved to `models/` before test evaluation.

---

### B3 — CVaR-Constrained PPO (Core Contribution)

**Type:** RL (CVaR-CMDP Lagrangian relaxation)

**CMDP formulation:**
```
max_θ  E[Σ γᵗ rₜ]
s.t.   CVaR_α(MaxDrawdown) ≤ d
```

**Lagrangian relaxation:**
```
max_θ min_λ  E[Σ γᵗ rₜ] − λ · (CVaR_α(DD) − d)
```

**Base reward (unmodified from B1):**
```
r_t = ΔPnL − φ · q² · Δt       (CjMmCriterion — identical to B1)
```

**λ update (per rollout, post-collection):**
```
violation = (CVaR_α(DD) − d) / max(d, 1e-8)
λ ← clip(λ + η · violation, 0, λ_max)
r'_t = r_t − λ · c_t
```

**Threshold calibration (training data only):**
```python
d = calibrate_cvar_threshold_windowed(
    env=get_doge_env(training_days_only=True),
    agent=b1_agent,
    n_steps=2048, cvar_alpha=0.2,
    n_windows=50, tighten=0.2
)
```
*Critically: d is calibrated from B1 rollouts on training days only.
Using test-day data to calibrate d would constitute look-ahead bias.*

**Warm-start:** Initialised from best B2 weights (lowest drawdown α that retains
positive Sharpe).

**Training:** Days 1–6, identical data to B1/B2

**Role in thesis:**
Core contribution. The adaptive λ enforces CVaR(DD) ≤ d without permanently modifying
the reward signal. When the constraint is violated, λ increases the penalty; when
satisfied, λ relaxes and the agent recovers PnL. CMDP strong duality guarantees the
Lagrangian saddle point equals the constrained optimum (Altman 1999 §3.2) given CVaR
coherence (Artzner et al. 1999) and Slater's condition (satisfied by the "don't trade"
policy, which achieves zero drawdown).

**Snellius note:** B3 training is the most compute-intensive job. Submit as a single
SLURM job with sufficient wall time (4–6 hours). Monitor λ convergence via TensorBoard
SSH tunnel (`ssh -L 6006:localhost:6006 hmalash@snellius.surf.nl "tensorboard --logdir
~/thesis/tb_logs/b3"`). If λ hits `lambda_max` and stays there, the constraint threshold
`d` may be too tight — re-calibrate with a larger `tighten` factor.

---

## 4. Evaluation Protocol

All four agents are evaluated on the **held-out test set (days 7–29) only**.
No test-day data is used during training or calibration.

### Metrics (following Falces Marín et al. 2022)

| Metric | Formula | Primary comparison |
|--------|---------|-------------------|
| Sharpe | mean(PnL) / std(PnL) | B0 vs B1 vs B3 |
| Sortino | mean(PnL) / std(PnL < 0) | B1 vs B3 |
| Max DD | max(peak − trough) over episode | B2 vs B3 (key result) |
| P&L-to-MAP | Ψ(T) / mean(\|q\|) | All four |
| Final PnL | Ψ(T) | All four |
| Mean \|q\| | mean absolute inventory | All four |

### Evaluation procedure
Each agent is rolled out on each of the 23 test days independently.
Per-day metrics are aggregated into mean ± std across test days.
The primary comparison table reports mean across all 23 test days.

---

## 5. Notebook Structure

### nb0_as_baseline.ipynb — B0 Evaluation

**Purpose:** Cross-validate γ on training days; evaluate fixed-γ B0 on test days.

**Section 1 — Setup**
```python
from procs.gym.experiment_config import ReplayExperimentConfig
from procs.gym.data_loader import load_multi_day
from procs.gym.calibration import calibrate_from_arrays, tune_gamma
from procs.gym.helpers.fast_rollout import fast_simulate

cfg = ReplayExperimentConfig()
daily_S, daily_dt, dates = load_multi_day(str(cfg.datasets_dir), pair=cfg.pair)

TRAIN_DAYS = 6
train_S, train_dt, train_dates = daily_S[:TRAIN_DAYS], daily_dt[:TRAIN_DAYS], dates[:TRAIN_DAYS]
test_S,  test_dt,  test_dates  = daily_S[TRAIN_DAYS:], daily_dt[TRAIN_DAYS:], dates[TRAIN_DAYS:]
```

**Section 2 — γ Cross-Validation on Training Days**
```python
# For each training day: estimate market parameters, then tune gamma
# Select gamma_star = mean of per-day optimal gammas across training days
# (or: run Optuna across the concatenated training set)
gamma_candidates = []
for S, dt in zip(train_S, train_dt):
    sigma, A, kappa = calibrate_from_arrays(S, dt, tick_size=cfg.tick_size)
    gamma_opt, _ = tune_gamma(S, dt, sigma, kappa, A, ...)
    gamma_candidates.append(gamma_opt)

gamma_fixed = float(np.mean(gamma_candidates))
print(f"Fixed gamma for B0: {gamma_fixed:.4f}")
```

**Section 3 — Test Evaluation (held-out days 7–29)**
```python
# Per-day: re-estimate sigma, A, kappa from that day's data
# Use gamma_fixed (no per-day tuning)
results_b0 = []
for S, dt, date in zip(test_S, test_dt, test_dates):
    sigma, A, kappa = calibrate_from_arrays(S, dt, tick_size=cfg.tick_size)
    stats = fast_simulate(S, dt, gamma=gamma_fixed, sigma=sigma, ...)
    results_b0.append({...})
```

**Section 4 — Results Table and Plots**
```python
df_b0 = pd.DataFrame(results_b0).set_index("Day")
# Mean ± std across test days for all metrics
# Save to results/b0_test_results.csv
```

---

### nb1_unconstrained_ppo.ipynb — B1 Training and Evaluation

**Purpose:** Train unconstrained PPO on training days; evaluate on test days.

**Section 1 — Environment Setup (Training)**
```python
from procs.gym.notebook_support import build_replay_env

# Training environment: days 1-6 only
train_env = build_replay_env(
    daily_S=train_S, daily_dt=train_dt,
    reward_cls=CjMmCriterion, phi=0.01,
    q_max=cfg.q_max, ...
)
train_vn = VecNormalize(train_env, norm_obs=True, norm_reward=True)
```

**Section 2 — PPO Training**
```python
model_b1 = PPO("MlpPolicy", train_vn,
    n_steps=2048, batch_size=512, n_epochs=10,
    learning_rate=3e-4, gamma=0.999,
    gae_lambda=0.95, clip_range=0.2, ent_coef=0.01,
    tensorboard_log="./tb_logs/b1/")

model_b1.learn(total_timesteps=cfg.train_timesteps)
model_b1.save("models/ppo_b1_doge.zip")
train_vn.save("models/vecnorm_b1.pkl")
```

**Section 3 — Test Evaluation (days 7–29)**
```python
# Separate eval environment: test days, same VecNormalize stats (frozen)
eval_env = build_replay_env(daily_S=test_S, daily_dt=test_dt, ...)
eval_vn = VecNormalize.load("models/vecnorm_b1.pkl", eval_env)
eval_vn.training = False   # freeze normalisation stats
eval_vn.norm_reward = False

agent_b1 = Sb3Agent(model_b1, vecnorm_env=eval_vn)
results_b1 = evaluate_agent(agent_b1, test_S, test_dt, test_dates, cfg)
# Save to results/b1_test_results.csv
```

**Section 4 — Training Diagnostics**
```python
# Load TensorBoard logs via EventAccumulator
# Plot: episode reward, entropy, value loss, explained variance
# Confirm convergence before proceeding to B2/B3
```

---

### nb2_dd_penalised_ppo.ipynb — B2 Training and Evaluation

**Purpose:** Train drawdown-penalised PPO across α sweep; evaluate on test days.

**Section 1 — α Sweep Training**
```python
ALPHAS = [0.1, 1.0, 10.0]

for alpha in ALPHAS:
    train_env = build_replay_env(
        reward_cls=CjMmDrawdownPenalty,
        phi=0.01, alpha=alpha, ...
    )
    model = PPO("MlpPolicy", VecNormalize(train_env), ...)
    model.learn(total_timesteps=cfg.train_timesteps)
    model.save(f"models/ppo_b2_alpha{alpha}_doge.zip")
```

**Section 2 — Test Evaluation (days 7–29) for Each α**
```python
# Evaluate each trained model on test days
# Collect: Sharpe, Sortino, MaxDD, PnL for each alpha
results_b2 = {}
for alpha in ALPHAS:
    model = PPO.load(f"models/ppo_b2_alpha{alpha}_doge.zip")
    results_b2[alpha] = evaluate_agent(agent, test_S, test_dt, test_dates, cfg)
```

**Section 3 — Pareto Frontier Plot**
```python
# X-axis: mean MaxDD across test days
# Y-axis: mean Final PnL across test days
# One point per alpha value
# This is the key B2 result: shows fixed PnL-drawdown tradeoff
```

**Section 4 — Select Best α for B3 Comparison**
```python
# Select alpha that minimises MaxDD while retaining positive Sharpe
# This becomes the primary B2 comparison point vs B3
best_alpha = ...
model_b2_best = PPO.load(f"models/ppo_b2_alpha{best_alpha}_doge.zip")
```

---

### nb3_cvar_constrained_ppo.ipynb — B3 Training and Evaluation

**Purpose:** Calibrate CVaR threshold from training data; train B3; evaluate on test days.

**Section 1 — CVaR Threshold Calibration (training days only)**
```python
from procs.gym.cvar_lagrangian import calibrate_cvar_threshold_windowed

# Use trained B1 agent, training environment only
train_env_eval = build_replay_env(daily_S=train_S, ...)
eval_vn_train = VecNormalize.load("models/vecnorm_b1.pkl", train_env_eval)
eval_vn_train.training = False

d = calibrate_cvar_threshold_windowed(
    env=eval_vn_train,
    agent=Sb3Agent(model_b1, eval_vn_train),
    n_steps=2048,
    cvar_alpha=0.2,
    n_windows=50,
    tighten=0.2          # d = CVaR_B1 * 0.8
)
print(f"CVaR threshold d = {d:.6f}")
# CRITICAL: this uses training days only — no test data touches this calibration
```

**Section 2 — B3 Training with Lagrangian Callback**
```python
from procs.gym.cvar_lagrangian import CVaRLagrangianCallback, DrawdownCostWrapper

train_env = DrawdownCostWrapper(build_replay_env(
    reward_cls=CjMmCriterion, ...  # base reward unchanged
))
train_vn = VecNormalize(train_env, ...)

model_b3 = PPO("MlpPolicy", train_vn, ...)

# Warm-start from B2 best weights
model_b3.set_parameters(model_b2_best.get_parameters())

cvar_callback = CVaRLagrangianCallback(
    cvar_alpha=0.2,
    threshold=d,
    lr_lambda=0.01,
    lambda_max=500.0,
    verbose=True
)

model_b3.learn(
    total_timesteps=cfg.train_timesteps,
    callback=cvar_callback
)
model_b3.save("models/ppo_b3_doge.zip")
```

**Section 3 — λ Convergence Diagnostics**
```python
# Plot lambda over training rollouts
# Confirm lambda stabilises (neither explodes to lambda_max nor collapses to 0)
# This is evidence that the constraint is active but satisfiable
```

**Section 4 — Test Evaluation (days 7–29)**
```python
eval_env = build_replay_env(daily_S=test_S, ...)
eval_vn = VecNormalize.load("models/vecnorm_b3.pkl", eval_env)
eval_vn.training = False

agent_b3 = Sb3Agent(model_b3, vecnorm_env=eval_vn)
results_b3 = evaluate_agent(agent_b3, test_S, test_dt, test_dates, cfg)
# Save to results/b3_test_results.csv
```

**Section 5 — B2 vs B3 Comparison (Central Result)**
```python
# Side-by-side bar charts: MaxDD, Sharpe, Final PnL per test day
# Summary table: mean ± std for B0, B1, B2 (best α), B3
# This is the main thesis result table
```

---

### nb4_comparison.ipynb — Four-Agent Summary

**Purpose:** Load all saved results CSVs and produce the final comparison tables and
plots, directly mirroring the format of Falces Marín et al. (2022) Tables 2–5.

**Section 1 — Load All Results**
```python
df_b0 = pd.read_csv("results/b0_test_results.csv").set_index("Day")
df_b1 = pd.read_csv("results/b1_test_results.csv").set_index("Day")
df_b2 = pd.read_csv("results/b2_test_results.csv").set_index("Day")  # best alpha
df_b3 = pd.read_csv("results/b3_test_results.csv").set_index("Day")
```

**Section 2 — One Table Per Metric (paper format)**

Following Falces Marín et al. (2022), produce **four separate tables**, one per metric.
Each table has:
- **Rows:** one per test day (Day 1 = Jan 7 through Day 23 = Jan 29)
- **Columns:** B0, B1, B2, B3
- **Bottom rows:** Days best, Median, Mean, Std. Dev.
- **Formatting:** best value per row in **bold**, second best underlined
- **Colour coding** (for thesis PDF via LaTeX): B3 values in green if better than B0,
  red if worse (mirroring how the paper colours Alpha-AS relative to Gen-AS)

Example table structure (Table X — Sharpe Ratio):

```
| Day        | B0    | B1    | B2    | B3    |
|------------|-------|-------|-------|-------|
| 2025-01-07 | x.xx  | x.xx  | x.xx  | x.xx  |
| 2025-01-08 | ...   | ...   | ...   | ...   |
| ...        |       |       |       |       |
| 2025-01-29 | x.xx  | x.xx  | x.xx  | x.xx  |
|------------|-------|-------|-------|-------|
| Days best  |  N    |  N    |  N    |  N    |
| Median     | x.xx  | x.xx  | x.xx  | x.xx  |
| Mean       | x.xx  | x.xx  | x.xx  | x.xx  |
| Std. Dev.  | x.xx  | x.xx  | x.xx  | x.xx  |
```

Replicate this structure for all four metrics: Sharpe, Sortino, Max DD, P&L-to-MAP.

```python
METRICS = {
    "Sharpe":      {"higher_better": True},
    "Sortino":     {"higher_better": True},
    "Max DD":      {"higher_better": False},
    "P&L-to-MAP":  {"higher_better": True},
}

def build_metric_table(dfs: dict, metric: str, higher_better: bool) -> pd.DataFrame:
    """
    dfs: dict of {agent_label: per-day DataFrame}
    Returns a table with one row per test day + summary rows,
    best value per day in bold (handled in LaTeX export).
    """
    table = pd.DataFrame({label: df[metric] for label, df in dfs.items()})
    table.index.name = "Day"

    # Summary rows
    days_best = table.apply(
        lambda row: row.idxmax() if higher_better else row.idxmin(), axis=1
    ).value_counts().reindex(table.columns, fill_value=0)

    summary = pd.DataFrame({
        label: {
            "Days best": days_best[label],
            "Median":    table[label].median(),
            "Mean":      table[label].mean(),
            "Std. Dev.": table[label].std(),
        }
        for label in table.columns
    }).T

    return pd.concat([table, summary.T])

dfs = {"B0": df_b0, "B1": df_b1, "B2": df_b2, "B3": df_b3}

for metric, props in METRICS.items():
    tbl = build_metric_table(dfs, metric, props["higher_better"])
    tbl.to_csv(f"results/table_{metric.replace(' ', '_').lower()}.csv")
    print(f"\n=== {metric} ===")
    print(tbl.to_string(float_format="%.4f"))
```

**Section 3 — Statistical Testing**

Following the paper, report a **Kruskal-Wallis test** across all four agents for each
metric (non-parametric, does not assume normality — appropriate for 23 test days):

```python
from scipy.stats import kruskal, mannwhitneyu

for metric in METRICS:
    groups = [df[metric].values for df in [df_b0, df_b1, df_b2, df_b3]]
    stat, p = kruskal(*groups)
    print(f"Kruskal-Wallis {metric}: H={stat:.2f}, p={p:.2e}")

# Pairwise Mann-Whitney: key comparison is B2 vs B3
for metric in METRICS:
    u, p = mannwhitneyu(df_b2[metric], df_b3[metric], alternative="two-sided")
    print(f"Mann-Whitney B2 vs B3 ({metric}): U={u:.1f}, p={p:.4f}")
```

**Section 4 — Key Plots**
```python
# Plot 1: Per-day MaxDD — B2 vs B3 (central result)
# Side-by-side bars for each test day; horizontal line at threshold d

# Plot 2: Per-day Sharpe — B0, B1, B3 line chart across 23 test days

# Plot 3: Pareto frontier — scatter plot
# X-axis: mean MaxDD; Y-axis: mean Final PnL
# Points: B0, B1, B2(α=0.1), B2(α=1.0), B2(α=10.0), B3
# B3 should dominate B2 at its drawdown level

# Plot 4: λ trajectory during B3 training
# X-axis: rollout index; Y-axis: lambda value
# Shows constraint becoming active then stabilising
```

**Section 5 — "Days Best" Summary Table**

Additional summary table following Falces Marín Table 6 — count of days each agent
achieved the best score across all metrics:

```
| Metric     | B0 | B1 | B2 | B3 |
|------------|----|----|----|----|
| Sharpe     |  N |  N |  N |  N |
| Sortino    |  N |  N |  N |  N |
| Max DD     |  N |  N |  N |  N |
| P&L-to-MAP |  N |  N |  N |  N |
```

---

## 6. Caveats and Limitations

The following should be addressed explicitly in the thesis discussion section:

**In-sample evaluation of B0 market parameters:**
σ, κ, A are re-estimated per test day from that day's data. This is methodologically
sound (these are market parameters, not preference parameters) and mirrors realistic
deployment. γ is fixed from training-day cross-validation.

**Training regime mismatch:**
Training days (Jan 1–6) are predominantly low-volatility. Test days include a
high-volatility episode (Jan 19–21). The RL agents have not encountered this regime
during training. This is a genuine generalisation test but means results on those days
should be interpreted cautiously.

**Thin CVaR estimation:**
CVaR at α=0.2 estimated from 50 windows of n_steps=2048 steps gives ~10 tail samples.
This is noisy. Report the threshold d with its standard error.

**Single asset, single month:**
Results are specific to DOGEUSDT in January 2025. Generalisation to other assets or
time periods is untested.

**Fill model optimism:**
The exponential Poisson fill model overestimates real fill rates compared to L2 order
book dynamics. All agents benefit equally from this, preserving relative comparisons,
but absolute PnL figures should not be taken as live-trading estimates.

**Six training days:**
PPO may not fully converge across all inventory/volatility states encountered in the
test set. Training curves should be inspected and reported in the appendix.

---

## 7. File Naming Conventions                   # B1 trained model
  vecnorm_b1.pkl                    # B1 VecNormalize stats
  ppo_b2_alpha0.1_doge.zip          # B2 α=0.1
  ppo_b2_alpha1.0_doge.zip          # B2 α=1.0
  ppo_b2_alpha10.0_doge.zip         # B2 α=10.0
  vecnorm_b2_alpha{X}.pkl           # B2 VecNormalize per alpha
  ppo_b3_doge.zip                   # B3 trained model
  vecnorm_b3.pkl                    # B3 VecNormalize stats

results/
  b0_test_results.csv               # B0 per-day test metrics
  b1_test_results.csv               # B1 per-day test metrics
  b2_alpha{X}_test_results.csv      # B2 per-day test metrics per alpha
  b3_test_results.csv               # B3 per-day test metrics
  cvar_threshold_d.txt              # Saved calibrated threshold value
  comparison_table.csv              # Final four-agent summary table
```

---

## 8. Compute Infrastructure (Snellius / MobaXterm)

All training and parallelised evaluation is run on **Snellius**, the Dutch national
supercomputer operated by SURF, accessed via **MobaXterm** SSH from a local Windows
machine.

### Connection and environment setup

```bash
# Connect via MobaXterm SSH session: snellius.surf.nl, username hmalash
# Then in the Snellius terminal:

module load 2023
module load Miniconda3/23.5.2-0
source $(conda info --base)/etc/profile.d/conda.sh
conda activate mysimenv
```

Run these four lines at the start of every Snellius session before executing any
Python scripts or submitting jobs.

### Partition

All jobs use the **`genoa`** partition (CPU-only). GPU nodes are not needed — the
PPO policy network operates on a 5-dimensional observation space and the compute
bottleneck is the CPU-bound Gymnasium simulation loop, not network operations.
Snellius bills a minimum of 24 CPUs per job on `genoa` regardless of `--cpus-per-task`.

### Code synchronisation

Code changes are pushed to GitHub locally and pulled on Snellius:
```bash
cd ~/thesis
git pull
```
Never edit code directly on Snellius without committing — changes will be lost if
the scratch filesystem is cleared or the session expires.

### Data location

Raw Tardis CSV files live on scratch (fast I/O, large quota):
```
/scratch-shared/hmalash/datasets/
```
This path is injected via the `DATA_DIR` environment variable, set in `~/.bashrc`:
```bash
export DATA_DIR=/scratch-shared/hmalash/datasets/
```

### Job submission pattern

Each training run is a separate SLURM job script submitted with `sbatch`. The general
pattern for a training job (`run.sh`):

```bash
#!/bin/bash
#SBATCH --partition=genoa
#SBATCH --cpus-per-task=24
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --job-name=<agent_name>
#SBATCH --output=/home/hmalash/thesis/logs/%j.out
#SBATCH --error=/home/hmalash/thesis/logs/%j.err

module load 2023
module load Miniconda3/23.5.2-0
source $(conda info --base)/etc/profile.d/conda.sh
conda activate mysimenv

export DATA_DIR=/scratch-shared/hmalash/datasets/
cd /home/hmalash/thesis

python train_<agent>.py
```

Parallelised evaluation over test days uses a job array:
```bash
sbatch --array=0-22%6 \
    --export=PROJECT_DIR=/home/hmalash/thesis,MANIFEST_PATH=...,CONDA_ENV=mysimenv \
    /home/hmalash/thesis/baseline_sweep.sh
```

The `%6` limits concurrent jobs to 6 to respect the account's `JobArrayTaskLimit`.

### Monitoring

```bash
squeue -u $USER                          # running/pending jobs
sacct -u hmalash --format=JobID,JobName,State,ExitCode,Reason -X | tail -20
cat /home/hmalash/thesis/logs/*.out      # stdout logs
cat /home/hmalash/thesis/logs/*.err      # stderr/error logs
```

For live TensorBoard during training, open a second MobaXterm terminal and run:
```bash
ssh -L 6006:localhost:6006 hmalash@snellius.surf.nl \
    "tensorboard --logdir ~/thesis/tb_logs"
```
Then open `http://localhost:6006` in a local browser.

### Retrieving results

After jobs complete, download results via MobaXterm's left-panel SFTP file browser
(navigate to `/gpfs/home2/hmalash/thesis/results/`) or via SCP from the local
WSL-Ubuntu terminal:
```bash
scp -r hmalash@snellius.surf.nl:/gpfs/home2/hmalash/thesis/results/ \
    "C:/Users/john-/Documents/Thesis_AI4T/results/"
scp -r hmalash@snellius.surf.nl:/gpfs/home2/hmalash/thesis/models/ \
    "C:/Users/john-/Documents/Thesis_AI4T/models/"
```

### Estimated wall times per agent

| Agent | Job type | Estimated wall time |
|-------|----------|-------------------|
| B0 γ calibration (train days) | Array, 6 jobs | ~1.5 hours total |
| B0 test evaluation (23 days) | Array, 23 jobs, %6 | ~4 hours total |
| B1 training | Single job | 2–4 hours |
| B2 α sweep (3 configs) | Array, 3 jobs | 2–4 hours each |
| B3 training | Single job | 4–6 hours |
| All test evaluations (B1–B3, 23 days) | Array per agent | ~1 hour each |

---

## 9. References

- Avellaneda, M. & Stoikov, S. (2008). High-frequency trading in a limit order book.
- Falces Marín et al. (2022). A reinforcement learning approach to improve the
  performance of the Avellaneda-Stoikov market-making algorithm.
- Altman, E. (1999). Constrained Markov Decision Processes. CRC Press.
- Artzner et al. (1999). Coherent measures of risk.
- Rockafellar & Uryasev (2000). Optimization of conditional value-at-risk.
- Ray et al. (2019). Benchmarking Safe Exploration in Deep RL. (Lagrangian formulation)
- Jerome et al. (2023). mbt-gym: Reinforcement learning for market-making.
