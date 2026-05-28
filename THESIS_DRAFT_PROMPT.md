# Thesis Draft Prompt: Formula-Based A-S Parameter RL With Legacy Placeholder Results

Use this markdown as the full context prompt for ChatGPT, Claude, or another writing assistant. The goal is to draft a thesis/paper chapter set that follows the **updated formula-based thesis structure**, while using the **current legacy/direct-depth result CSVs only as temporary placeholder results** until the formula-based B1-B3 models are fully rerun.

## Non-Negotiable Writing Rules

- Write the thesis as if the final methodological contribution is **formula-based reinforcement learning for Avellaneda-Stoikov market making**, where RL chooses interpretable A-S control parameters rather than raw bid/ask quote depths.
- Use the numerical results below only as **temporary placeholder results from the older direct quote-depth PPO implementation**.
- Do **not** claim that the old B1-B3 numbers are final evidence for the formula-based implementation.
- Whenever discussing B1-B3 results, label them as: `legacy direct-depth PPO placeholder results` or `preliminary legacy results`.
- The ablation study and SHAP analysis are planned components. Explain the design and expected interpretive value, but leave result values open as `TODO: insert ablation/SHAP results after rerun`.
- Hummingbot backtesting must be promised carefully as a **planned secondary validation / deployment-oriented backtest**, not as completed empirical evidence.
- Use citations for all methodological claims. Do not invent citations. If a citation is uncertain, mark it as `citation needed`.
- Keep the writing academically cautious. The central thesis should be defensible even if the final formula-based RL models do not outperform B0 on every metric.

## Suggested Working Title

**Risk-Aware Formula-Based Reinforcement Learning for Avellaneda-Stoikov Market Making on Cryptocurrency Limit Order Book Data**

Alternative shorter title:

**Risk-Aware Reinforcement Learning Extensions of the Avellaneda-Stoikov Market-Making Framework**

## Central Research Question

Can reinforcement learning improve or risk-adjust the Avellaneda-Stoikov market-making framework on cryptocurrency limit order book data by learning adaptive A-S control parameters, while preserving the interpretability and structure of the analytical model?

## Core Thesis Position

The thesis should not present RL as a black-box replacement for Avellaneda-Stoikov. Instead, it should present RL as a **parameter-control layer** on top of A-S. This is academically cleaner because the learned policy remains tied to the theoretical structure of market making: reservation price, optimal spread, inventory aversion, and skew.

The intended final comparison is:

- `Plain AS`: simple analytical A-S without learning or risk-objective extensions.
- `B0`: analytical A-S baseline with train-only calibration/tuning.
- `B1`: formula-based PPO agent selecting A-S parameters with a standard PnL/inventory reward.
- `B2`: formula-based PPO agent selecting A-S parameters with a drawdown-penalized reward.
- `B3`: formula-based PPO agent selecting A-S parameters under a CVaR-style drawdown constraint.

The temporary result tables currently available for B1-B3 are from the older architecture, where PPO directly output quote depths. These are included only so the first thesis draft has filled result sections.

## Updated Methodological Structure

### Analytical A-S Foundation

Use Avellaneda and Stoikov's market-making model as the analytical foundation. In the finite-horizon A-S approximation, the reservation price and half-spread are:

```text
r_t = S_t - q_t * gamma_t * sigma^2 * (T - t)

h_t = 0.5 * gamma_t * sigma^2 * (T - t)
      + (1 / gamma_t) * log(1 + gamma_t / kappa)
```

where:

- `S_t` is the mid-price;
- `q_t` is inventory;
- `gamma_t` is risk aversion;
- `sigma` is mid-price volatility;
- `kappa` controls the sensitivity of fill intensity to quote distance;
- `T - t` is remaining episode time.

The formula-based RL extension uses the policy only to choose adaptive A-S controls:

```text
pi_theta(x_t) -> [u_gamma_t, u_skew_t]

gamma_t = exp(log(gamma_min) + ((u_gamma_t + 1) / 2) * (log(gamma_max) - log(gamma_min)))
skew_t = u_skew_t * skew_ticks_max * tick_size

center_t = r_t + skew_t
bid_t = floor((center_t - h_t) / tick_size) * tick_size
ask_t = ceil((center_t + h_t) / tick_size) * tick_size
```

This preserves the A-S structure: the policy does not directly choose arbitrary quotes; it adapts risk aversion and quote skew.

### Difference From the Legacy Direct-Depth PPO Models

The older implementation used:

```text
pi_theta(x_t) -> [delta_bid_t, delta_ask_t]
bid_t = S_t - delta_bid_t
ask_t = S_t + delta_ask_t
```

This is a direct continuous-control quoting policy. It is more flexible, but less interpretable and further from the A-S model. The updated formula-based approach is closer to Falces Marin et al.'s Alpha-AS idea because the RL policy adjusts A-S parameters rather than replacing the A-S quote formula.

### Relationship to Falces Marin et al.

Falces Marin et al. propose Alpha-AS models in which a Double DQN agent modifies A-S parameters, rather than directly setting quotes. Their Alpha-AS variants adapt risk aversion and skew over fixed action windows, and the A-S formula remains responsible for quote construction.

The thesis should position itself as a related but distinct contribution:

- Falces Marin et al. use Double DQN with a discrete parameter grid.
- This thesis uses PPO with continuous parameter actions.
- Falces Marin et al. focus on improving A-S performance through RL parameter control.
- This thesis focuses on **risk-aware A-S parameter control**, comparing unconstrained, drawdown-penalized, and CVaR-constrained objectives.

## Why PPO Instead of DQN or A2C

Use the following argument, but support it with citations.

PPO is chosen because the final formula-based model has a continuous bounded action space: the policy chooses continuous `gamma_t` and `skew_t` controls. DQN is naturally suited to discrete action spaces and was appropriate for Falces Marin et al.'s Alpha-AS grid, but using DQN here would require discretizing the A-S parameter space. That would change the research question by mixing three changes at once: action discretization, algorithm choice, and risk objective.

A2C can support continuous actor-critic learning, but PPO is generally more stable for noisy policy-gradient training because it uses a clipped surrogate objective that discourages excessively large policy updates. In this thesis, that stability is valuable because market-making rewards are noisy, path-dependent, and sensitive to inventory and fill simulation.

The CVaR extension does not mathematically require PPO. A CVaR or CMDP-style objective can be combined with other RL algorithms. However, the current implementation is PPO-based because the Lagrangian/CVaR callback operates on PPO rollouts, and PPO keeps the comparison between B1, B2, and B3 clean: all three models share the same policy class and action structure, while only the risk objective changes.

## Dataset And Split

Use DOGE cryptocurrency limit order book / replay data. The intended split is:

- Training/calibration days: `2025-01-01` through `2025-01-06`.
- Test days: `2025-01-07` through `2025-01-29`.
- Number of test days in current result CSVs: `23`.

The thesis should emphasize strict train/test separation:

- A-S market parameters used for final evaluation should be estimated from the training period, not re-estimated on each test day.
- RL policies should train only on the training days.
- Feature normalization should be fitted on training data only.
- Test days should be used only for out-of-sample evaluation.

## Models To Describe

### Plain AS

A simple analytical Avellaneda-Stoikov strategy without learning and without the risk-aware RL extensions. Use it as a transparent benchmark showing what the base formula does on DOGE data.

### B0: Analytical A-S Baseline

B0 is the stronger analytical baseline. It uses the A-S formula with parameters calibrated/tuned on the training set. Current fixed gamma file reports:

```text
gamma_fixed = 0.975143028728294
```

Make clear that B0 is not a neural policy. It is formula-based and interpretable.

### B1: Formula-Based PPO A-S Parameter Control

B1 should be described as:

```text
pi_theta(x_t) -> gamma_t, skew_t -> A-S formula -> bid_t, ask_t
```

Its objective is standard market-making reward, such as PnL change with inventory penalty.

### B2: Drawdown-Penalized Formula-Based PPO

B2 uses the same formula-based action structure as B1, but modifies the reward to penalize drawdown:

```text
R_t^B2 = R_t^B1 - alpha * DD_t
```

where `DD_t` is current drawdown from the running portfolio-value peak. The current placeholder B2 result uses:

```text
best_alpha = 1.0
```

### B3: CVaR-Constrained Formula-Based PPO

B3 keeps the same policy/action structure but frames risk control as a constrained objective:

```text
maximize_theta E[return]
subject to CVaR_beta(MaxDrawdown) <= d
```

A Lagrangian form can be written as:

```text
L(theta, lambda) = J(theta) - lambda * (CVaR_beta(MaxDrawdown_theta) - d)
```

This model should be motivated by the fact that drawdown risk is path-dependent and tail-sensitive; CVaR/expected shortfall is more informative for tail losses than only reporting average drawdown.

## Feature Engineering Plan

The updated formula-based models should use online, no-leakage features. Recommended feature set:

- normalized inventory;
- time remaining fraction;
- relative/log mid-price;
- rolling volatility estimated from past returns only;
- short-horizon return momentum from past observations only;
- bid-ask spread in ticks;
- limit order book imbalance if quantity columns are reliable.

Important wording:

- These are not future-looking features.
- Feature normalization must be fitted on training days only.
- Feature engineering is intended to improve state representation, not to leak test information.

## Ablation Study Plan

The ablation study should be described as planned or pending until rerun. Suggested design:

- `Base`: normalized inventory, time remaining, mid-price/relative mid-price.
- `Base + volatility`: base features plus rolling volatility.
- `Full`: base features plus volatility, momentum, spread, and order-book imbalance.

Use an internal training validation split:

- Train candidate feature sets on days 1-4.
- Validate on days 5-6.
- Select the feature set based on validation performance and risk stability.
- Retrain final B1-B3 on days 1-6.
- Evaluate only on days 7-29.

Do not fabricate ablation outcomes. Use placeholders:

```text
TODO: Insert ablation table after formula-based rerun.
TODO: Discuss whether engineered microstructure features improve out-of-sample risk-adjusted returns.
```

## SHAP / Shapley Interpretability Plan

Use SHAP as an interpretability method for the trained actor, not as causal proof. The actor can be wrapped as:

```text
f(x_t) -> [mean_u_gamma_t, mean_u_skew_t]
```

Then SHAP can explain which features contribute to the policy's chosen risk-aversion and skew actions. The thesis should be careful:

- SHAP explains model outputs, not trading profitability.
- SHAP values do not prove causal market effects.
- SHAP should be paired with ablation, because ablation tests whether feature groups actually matter for performance.

Use placeholders:

```text
TODO: Insert SHAP feature-importance plots after final B1-B3 training.
TODO: Interpret gamma-action and skew-action explanations separately.
```

## Hummingbot Backtesting Promise

Use cautious wording. Do not claim completed Hummingbot validation unless it has been run.

Recommended wording:

> As a deployment-oriented extension, the formula-based policy can be wrapped in a Hummingbot-compatible controller. This is not treated as the primary empirical evidence in the thesis. Instead, it is proposed as a secondary validation step, because the core thesis evaluates policies in a controlled replay simulator. The formula-based structure makes Hummingbot integration more practical than direct-depth neural quoting: the deployed strategy still computes interpretable A-S quotes, with RL only adapting bounded parameter controls.

Mention required safety and engineering checks:

- exchange tick-size and lot-size rounding;
- minimum spread constraints;
- maker/taker fees;
- latency and order refresh logic;
- inventory caps;
- cancellation logic;
- kill switch / stop-loss constraints;
- quote parity between offline simulator and Hummingbot adapter.

## Metrics

Use the following metrics in results chapters:

- Sharpe ratio: risk-adjusted average return using return volatility.
- Sortino ratio: downside-risk-adjusted return.
- Maximum drawdown: worst peak-to-trough decline in portfolio value.
- PnL-to-MAP: final PnL relative to mean absolute position; interpret as efficiency of inventory usage.
- Final PnL: final mark-to-market portfolio value change.
- Mean absolute inventory: average inventory exposure.
- Near-cap fraction: fraction of time inventory is close to the inventory cap.

Be careful with directionality:

- Higher is better: Sharpe, Sortino, PnL-to-MAP, Final PnL.
- Lower is better: Max DD, Mean |q| if inventory exposure is treated as risk, Near Cap Fraction.

## Extracted Placeholder Results From Current CSVs

These are current local results extracted from:

- `results/plain_as_test_results.csv`
- `results/b0_test_results.csv`
- `results/b1_test_results.csv`
- `results/b2_test_results.csv`
- `results/b3_test_results.csv`

Again: B1-B3 are **legacy direct-depth PPO placeholder results**, not final formula-based PPO results.

### Mean Test Metrics Across 23 Days

| Model    |   Sharpe |   Sortino |   Max DD |   P&L-to-MAP |   Final PnL |   Mean |q| |   Near Cap Fraction |
|:---------|---------:|----------:|---------:|-------------:|------------:|-----------:|--------------------:|
| Plain AS |   0.0239 |    0.0107 |   0.0451 |       0.36   |      1.0342 |     2.8752 |              0.0546 |
| B0       |   0.0582 |    0.03   |   0.01   |       1.661  |      1.4133 |     0.7598 |              0.0032 |
| B1       |   0.0134 |    0.0051 |   0.0079 |       0.373  |      0.1132 |     0.3762 |              0      |
| B2       |   0.0203 |    0.006  |   0.0046 |       2.4719 |      0.1209 |     0.3149 |              0      |
| B3       |   0.0232 |    0.0061 |   0.0043 |       3.5853 |      0.1304 |     0.251  |              0      |

### Median Test Metrics Across 23 Days

| Model    |   Sharpe |   Sortino |   Max DD |   P&L-to-MAP |   Final PnL |   Mean |q| |   Near Cap Fraction |
|:---------|---------:|----------:|---------:|-------------:|------------:|-----------:|--------------------:|
| Plain AS |   0.0222 |    0.0109 |   0.036  |       0.3602 |      1.0327 |     2.8719 |              0.0548 |
| B0       |   0.0609 |    0.0233 |   0.0078 |       1.1501 |      0.9163 |     0.8577 |              0.003  |
| B1       |   0.0133 |    0.0047 |   0.0071 |       0.2913 |      0.1071 |     0.3884 |              0      |
| B2       |   0.0183 |    0.006  |   0.0039 |       0.3736 |      0.1156 |     0.3247 |              0      |
| B3       |   0.023  |    0.006  |   0.0031 |       1.6106 |      0.1599 |     0.0993 |              0      |

### Days-Best Counts Across B0-B3

| Metric            | Direction        |   B0 |   B1 |   B2 |   B3 |
|:------------------|:-----------------|-----:|-----:|-----:|-----:|
| Sharpe            | higher is better |   18 |    0 |    3 |    2 |
| Sortino           | higher is better |   18 |    0 |    5 |    0 |
| Max DD            | lower is better  |    4 |    5 |    8 |    6 |
| P&L-to-MAP        | higher is better |   11 |    0 |    6 |    6 |
| Final PnL         | higher is better |   19 |    1 |    3 |    0 |
| Mean |q|          | lower is better  |    0 |    8 |    6 |    9 |
| Near Cap Fraction | lower is better  |    0 |   23 |    0 |    0 |

### B2 Alpha Sweep Placeholder Summary

|   alpha |   mean_sharpe |   mean_sortino |   mean_max_dd |   mean_final_pnl |   mean_abs_inventory |
|--------:|--------------:|---------------:|--------------:|-----------------:|---------------------:|
|  0.1000 |        0.0190 |         0.0051 |        0.0055 |           0.1207 |               0.3122 |
|  1.0000 |        0.0203 |         0.0060 |        0.0046 |           0.1209 |               0.3149 |
| 10.0000 |        0.0097 |         0.0041 |        0.0117 |           0.0899 |               0.5061 |

## Placeholder Result Interpretation

Use this interpretation for the first draft, but repeatedly mark it as preliminary/legacy.

The analytical B0 baseline currently dominates the legacy B1-B3 comparison on mean Sharpe, Sortino, and final PnL. B0 wins 18 of 23 test days by Sharpe and 18 of 23 test days by Sortino. It also wins 19 of 23 test days by final PnL. This suggests that the older direct-depth PPO policies did not learn a sufficiently profitable quoting rule relative to the analytical A-S baseline.

However, the legacy RL models show a different risk profile. B2 and B3 reduce mean maximum drawdown relative to B0, and B3 has the lowest mean maximum drawdown among B1-B3. B3 also has the lowest mean absolute inventory among B1-B3. This is consistent with the intended risk-aware objectives: risk penalties and constraints can reduce exposure and drawdown, but may sacrifice profitability.

The preliminary conclusion should be nuanced:

- The old direct-depth PPO architecture underperforms B0 in profitability and risk-adjusted return.
- Risk-aware objectives appear to shift the learned policies toward lower inventory exposure and lower drawdown.
- This motivates the formula-based redesign: rather than asking PPO to learn quotes from scratch, PPO should control interpretable A-S parameters while the A-S formula constructs quotes.
- The final empirical question is whether formula-based PPO can retain the risk discipline of B2/B3 while closing the profitability gap to B0.

## Suggested Results Chapter Structure

1. Introduce all metrics and test-day protocol.
2. Present Plain AS and B0 analytical baseline results.
3. Present legacy placeholder B1-B3 results in a clearly labeled subsection.
4. Explain why the placeholder results motivate formula-based parameter control.
5. Leave a marked subsection for final formula-based B1-B3 results.
6. Leave marked subsections for ablation and SHAP.
7. End with a careful discussion of Hummingbot as future/secondary validation.

Example wording:

> The numerical results in this draft correspond to an earlier direct-depth PPO implementation. They are included as placeholders to illustrate the planned reporting structure. The final thesis will replace these tables with the formula-based PPO results, where the policy selects A-S risk-aversion and skew parameters rather than raw quote depths.

## Expected Final Claims After Formula-Based Rerun

Do not state these as completed findings yet. They are hypotheses / intended evaluation criteria:

- Formula-based PPO should be more interpretable than direct-depth PPO because quotes remain generated by A-S equations.
- B1 tests whether adaptive A-S parameters improve average performance.
- B2 tests whether drawdown penalties can reduce realized drawdown and inventory exposure.
- B3 tests whether explicit tail-risk constraints can control adverse drawdown outcomes.
- Ablation tests whether engineered microstructure features improve out-of-sample policy quality.
- SHAP helps interpret which state variables drive adaptive risk aversion and skew.
- Hummingbot backtesting assesses whether the formula-based policy can be translated into a more realistic execution framework.

## Citation Anchors To Use

Use these sources as anchors. Verify formatting in the final bibliography.

- Avellaneda, M., & Stoikov, S. (2008). **High-frequency trading in a limit order book**. Quantitative Finance, 8(3), 217-224. DOI: `10.1080/14697680701381228`. URL: https://www.tandfonline.com/doi/abs/10.1080/14697680701381228
- Falces Marin, J., Diaz Pardo de Vera, D., & Lopez Gonzalo, E. (2022). **A reinforcement learning approach to improve the performance of the Avellaneda-Stoikov market-making algorithm**. PLOS ONE, 17(12), e0277042. URL: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0277042
- Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). **Proximal Policy Optimization Algorithms**. arXiv:1707.06347. URL: https://arxiv.org/abs/1707.06347
- Mnih, V., et al. (2015). **Human-level control through deep reinforcement learning**. Nature, 518, 529-533. URL: https://www.nature.com/articles/nature14236
- Mnih, V., et al. (2016). **Asynchronous Methods for Deep Reinforcement Learning**. ICML/arXiv:1602.01783. URL: https://arxiv.org/abs/1602.01783
- Altman, E. (1999). **Constrained Markov Decision Processes**. Chapman & Hall/CRC. URL: https://openlibrary.org/books/OL97592M/Constrained_Markov_decision_processes
- Achiam, J., Held, D., Tamar, A., & Abbeel, P. (2017). **Constrained Policy Optimization**. arXiv:1705.10528. URL: https://arxiv.org/abs/1705.10528
- Artzner, P., Delbaen, F., Eber, J.-M., & Heath, D. (1999). **Coherent Measures of Risk**. Mathematical Finance, 9(3), 203-228. URL: https://people.math.ethz.ch/~delbaen/ftp/preprints/CoherentMF.pdf
- Rockafellar, R. T., & Uryasev, S. (2000). **Optimization of Conditional Value-at-Risk**. Journal of Risk, 3, 21-41. DOI: `10.21314/JOR.2000.038`. URL: https://colab.ws/articles/10.21314%2Fjor.2000.038
- Lundberg, S. M., & Lee, S.-I. (2017). **A Unified Approach to Interpreting Model Predictions**. NeurIPS. URL: https://papers.neurips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions
- Cartea, A., Jaimungal, S., & Penalva, J. (2015). **Algorithmic and High-Frequency Trading**. Cambridge University Press. URL: https://books.google.co.uk/books?vid=ISBN9781107091146
- Stable-Baselines3 documentation for practical PPO implementation details and action-space support. URL: https://stable-baselines3.readthedocs.io/en/v2.5.0/modules/ppo.html

## Claims That Need Citations Or Careful Framing

Use this section to avoid overclaiming.

- It is valid to say A-S is a classical analytical market-making benchmark.
- It is valid to say Falces Marin et al. use RL to adapt A-S parameters rather than replacing the quote formula.
- It is valid to say DQN is naturally associated with discrete action-value learning, while PPO directly supports continuous stochastic policies.
- It is valid to say CVaR/expected shortfall is tail-risk focused and commonly used in risk management.
- It is valid to say SHAP explains model outputs through feature attribution, but not causal market mechanisms.
- It is not valid to say the formula-based B1-B3 models outperform B0 until the final formula-based rerun is complete.
- It is not valid to say Hummingbot backtesting confirms deployability until it has been run.

## Suggested Abstract Draft Skeleton

This thesis studies risk-aware reinforcement learning extensions of the Avellaneda-Stoikov market-making framework on cryptocurrency limit order book data. Rather than replacing the analytical model with a black-box quoting policy, the proposed final approach uses PPO to adapt interpretable A-S control parameters, including risk aversion and quote skew. Three RL variants are compared: an unconstrained parameter-control policy, a drawdown-penalized policy, and a CVaR-constrained policy. The study evaluates these methods against analytical A-S baselines using out-of-sample DOGE market data and metrics including Sharpe ratio, Sortino ratio, maximum drawdown, final PnL, and inventory exposure. Preliminary placeholder results from an earlier direct-depth PPO implementation suggest that unconstrained black-box quote learning underperforms the analytical baseline in profitability, while risk-aware objectives reduce drawdown and inventory exposure. These preliminary findings motivate the formula-based redesign. Planned ablation and SHAP analyses will assess the contribution and interpretability of engineered microstructure features, while Hummingbot backtesting is proposed as a secondary validation path for deployment realism.

## Suggested Conclusion Draft Skeleton

The thesis should conclude cautiously. The strongest defensible conclusion is not necessarily that RL dominates A-S, but that risk-aware RL can be integrated with A-S in an interpretable way. The legacy direct-depth placeholder results show that learning quotes directly is difficult and may underperform a calibrated analytical baseline. This motivates the formula-based approach: retain A-S quote construction, but learn adaptive parameter controls. If final formula-based results improve risk-adjusted performance, the thesis can claim empirical support for this structure. If they do not, the thesis can still contribute a rigorous negative or mixed result: A-S remains a strong benchmark, and risk-aware RL primarily trades profitability for drawdown/inventory control under the tested DOGE replay conditions.

## TODO Markers For The Final Thesis

- `TODO: Replace legacy direct-depth PPO result tables with final formula-based B1-B3 rerun results.`
- `TODO: Add final formula-based comparison table.`
- `TODO: Add ablation study results.`
- `TODO: Add SHAP plots and interpretation.`
- `TODO: Add Hummingbot backtest results only if actually completed.`
- `TODO: Verify all bibliography entries and citation formatting.`
