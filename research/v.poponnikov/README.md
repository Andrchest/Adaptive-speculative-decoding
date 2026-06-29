# v.poponnikov Research Area

## Research Direction

This research track studies stochastic methods for choosing the speculative
draft length `k` dynamically during generation. The goal is to improve the
speed and stability of speculative decoding by increasing `k` when the drafter
is locally reliable and shrinking `k` when long drafts are likely to be
rejected by the target model.

The work focuses on two methods:

1. `EpistemicConsensusK`: choose `k` from agreement between several stochastic
   draft trajectories.
2. `LatentRegimeK`: choose `k` from an online hidden-regime model with
   change-point resets.

## Motivation

Speculative decoding trades off drafter compute against target verification
efficiency. A fixed draft length can be suboptimal because the best `k` changes
across prompts, token positions, and local text regimes. Small `k` limits
speedup, while large `k` can waste drafter work when most draft tokens are
rejected.

This track treats `k` selection as an online stochastic control problem. Each
generation step observes local signals such as acceptance rate, rejection
position, drafter uncertainty, and drafter-target disagreement, then samples the
next `k` from an adaptive distribution.

## Hypotheses

- Stochastic drafter self-consensus can estimate local uncertainty well enough
  to choose safer draft lengths than a fixed `k`.
- Online adaptation toward a target acceptance rate can stabilize the tradeoff
  between tokens per second and rejected draft work.
- Hidden generation regimes, such as easy text, normal text, reasoning/code,
  and transition points, can explain changes in the optimal draft length.
- A change-point reset can prevent long rejected drafts immediately after topic
  or format shifts.

## Method 1: Epistemic Consensus K

`EpistemicConsensusK` runs `M` stochastic drafter trajectories up to `K_max`.
The stochasticity can come from sampling temperature, small logit noise, or
dropout if available. For each draft position, the controller estimates whether
the trajectories agree.

Signals:

- `consensus_j`: fraction of trajectories that select the majority token at
  position `j`.
- `logprob_variance_j`: variance of the majority token log probability across
  stochastic runs.
- `margin_j`: gap between the top token and the second-best token.
- `acceptance_rate_t`: verified acceptance rate from the previous step.

Position score:

```text
score_j = lambda_c * consensus_j
        + lambda_m * sigmoid(margin_j / tau_m)
        - lambda_u * logprob_variance_j
```

Continuation probability:

```text
continue_prob_j = sigmoid((score_j - theta_t) / tau_k)
```

The controller samples from left to right and stops at the first failed
continuation draw. After target verification, it updates the caution threshold:

```text
theta_{t+1} = clip(theta_t + eta * (rho - acceptance_rate_t), theta_min, theta_max)
```

If the observed acceptance rate falls below the target `rho`, future drafts
become shorter. If acceptance is consistently high, future drafts can grow.

## Method 2: Latent Regime / Change-Point K

`LatentRegimeK` models generation as a sequence of hidden regimes. Each regime
has its own distribution over `k`, and the controller updates regime
probabilities after each target verification.

Initial regimes:

| Regime | Expected behavior |
| --- | --- |
| `easy` | Stable text where larger `k` is usually safe. |
| `normal` | General text where medium `k` is preferred. |
| `hard` | Reasoning, math, or code where smaller `k` may reduce waste. |
| `transition` | Topic or format shifts where `k` should reset near minimum. |

Features after each verification:

- `acceptance_rate_t`
- `accepted_tokens_t`
- `selected_k_t`
- drafter entropy
- drafter-target disagreement
- token class, such as text, code, newline, number, markdown, or math
- first rejection position

The controller maintains a posterior over regimes and samples:

```text
k_t ~ TruncatedPoisson(lambda_t, K_min, K_max)
lambda_t = (1 - change_point_t) * lambda_regime + change_point_t * lambda_min
```

When the change-point probability is high, `lambda_t` moves toward `lambda_min`
and the controller sharply reduces `k`.

## Implementation Plan

1. Implement `EpistemicConsensusK` as a research adaptive controller. Done.
2. Add a research experiment under `research/v.poponnikov/experiments/`. Done.
3. Compare against `01_baseline` and `08_+speedup_adapt`.
4. Add unit tests for sampling, threshold adaptation, and bounds on `k`. Done.
5. Add lightweight smoke experiments with tiny models.
6. Record metrics and notes in `research/v.poponnikov/results/`.
7. Implement `LatentRegimeK` as the second stochastic controller. Done.

## Metrics

Primary metrics:

- `tokens_per_sec`
- `acceptance_rate`
- `avg_accepted_tokens`
- `avg_draft_length`
- `wall_time_total_s`

Research-specific metrics:

- mean selected `k`
- distribution of selected `k`
- rejection position histogram
- consensus score statistics
- threshold trajectory for `EpistemicConsensusK`
- regime posterior entropy for `LatentRegimeK`
- change-point reset frequency

## Experiment Commands

Fast iteration:

```bash
python src/main.py --research --tiny -n 5 --max-new-tokens 32 --no-mlflow
```

Single experiment:

```bash
python src/main.py --experiment stochastic_consensus_k --tiny -n 5 --max-new-tokens 32 --no-mlflow
```

Reference baselines:

```bash
python src/main.py --experiment 01_baseline --tiny -n 5 --max-new-tokens 32 --no-mlflow
python src/main.py --experiment 08_+speedup_adapt --tiny -n 5 --max-new-tokens 32 --no-mlflow
```

Comparison with plots:

Notebook workflow for online IDEs without terminal access:

1. Open `research/v.poponnikov/notebooks/dynamic_k_comparison.ipynb`.
2. Run the cells from top to bottom.
3. On a fresh Python 3.10 online image, run the dependency install cell once,
   then restart the notebook kernel and set `INSTALL_DEPENDENCIES = False`.
4. Keep the tiny smoke run enabled first.
5. After the smoke run succeeds, set `RUN_REAL = True` in the real Qwen
   comparison cell and run it.

Command-line workflow:

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe research\v.poponnikov\notebooks\dynamic_k_comparison.py `
  --tiny `
  --samples 5 `
  --max-new-tokens 32 `
  --device cuda
```

This runs `01_baseline`, `08_+speedup_adapt`, `stochastic_consensus_k`, and
`latent_regime_k` in one comparison pass. Results are written to
`research/v.poponnikov/results/dynamic_k_comparison/`, including
`dynamic_k_comparison.csv`. Plots are written to
`research/v.poponnikov/plots/dynamic_k_comparison/`.

The notebook workflow keeps smoke and real runs in separate folders:

- `research/v.poponnikov/results/smoke/`
- `research/v.poponnikov/results/real/`
- `research/v.poponnikov/plots/smoke/`
- `research/v.poponnikov/plots/real/`

To regenerate plots from existing JSON results without rerunning models:

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe research\v.poponnikov\notebooks\dynamic_k_comparison.py --plot-only
```

## Preliminary Results

Smoke comparison on `gsm8k`, 5 samples, 32 max new tokens, with
`facebook/opt-125m` as drafter and `facebook/opt-350m` as target:

| Experiment | Tokens/sec | Acceptance rate | Avg accepted | Avg draft length | Wall time |
| --- | ---: | ---: | ---: | ---: | ---: |
| `01_baseline` | 6.93 | 21.29% | 1.09 | 5.00 | 10.24 s |
| `08_+speedup_adapt` | 7.73 | 23.00% | 1.21 | 5.47 | 11.89 s |
| `stochastic_consensus_k` | 1.93 | 57.99% | 0.63 | 1.17 | 26.91 s |
| `latent_regime_k` | 10.36 | 54.97% | 1.78 | 3.51 | 10.81 s |

Interpretation:

- `LatentRegimeK` is the strongest smoke result. It improves throughput over
  the fixed baseline and speedup-adaptive baseline while also raising
  acceptance rate. It selected a moderate mean `k` of 3.51 and used the full
  range from 1 to 8, which suggests the regime posterior is doing useful
  online adaptation instead of collapsing to a fixed value.
- `EpistemicConsensusK` is not competitive in the current configuration. Its
  acceptance rate is high because it collapses to very short drafts, with mean
  selected `k` 1.17 and 74 of 83 selections at `k = 1`. The method also pays
  for 4 stochastic drafter trajectories before each actual draft, so throughput
  falls sharply.
- The smoke run is too small to prove the regime method is generally better.
  It is enough to show that the implementation works and that the consensus
  method needs retuning before it is a useful baseline.

Next result needed:

- Repeat the smoke run if the implementation changes, but do not use it as the
  main research conclusion because the drafter-target size gap is small.

Real-model comparison on `gsm8k`, 50 samples, 128 max new tokens, with
`Qwen/Qwen2.5-0.5B-Instruct` as drafter and
`Qwen/Qwen2.5-7B-Instruct` as target:

| Experiment | Tokens/sec | Acceptance rate | Avg accepted | Avg draft length | Wall time |
| --- | ---: | ---: | ---: | ---: | ---: |
| `01_baseline` | 6.21 | 38.88% | 1.86 | 5.00 | 680.84 s |
| `08_+speedup_adapt` | 5.04 | 40.24% | 1.60 | 4.22 | 797.10 s |
| `stochastic_consensus_k` | 0.72 | 68.74% | 0.70 | 1.03 | 3691.99 s |
| `latent_regime_k` | 4.52 | 51.41% | 1.16 | 2.33 | 766.38 s |

Real-run interpretation:

- The fixed baseline is the strongest throughput result in the Qwen run. It
  reaches 6.21 tokens/sec, while `latent_regime_k` reaches 4.52 tokens/sec.
  This means the current regime controller improves acceptance quality but is
  too conservative to improve wall-clock speed.
- `LatentRegimeK` is still promising as a control signal. It raises acceptance
  from 38.88% to 51.41%, but it reduces mean selected `k` from the fixed value
  of 5.00 to 2.33. The result suggests the posterior is detecting harder
  regions, but the lambda/reward update currently shrinks drafts too much.
- `EpistemicConsensusK` is not viable in its current form. It achieves high
  acceptance only by collapsing almost always to `k = 1`:
  3708 of 3774 selections are `k = 1`. The extra 4 drafter trajectories make
  it far slower than every other method.
- `08_+speedup_adapt` also underperforms the fixed baseline in this run. The
  broader lesson is that higher acceptance alone is not sufficient; the chosen
  `k` must be large enough to amortize target verification and any controller
  overhead.

Next tuning direction:

- Make `LatentRegimeK` less conservative by slowing lambda shrinkage, raising
  the lower bound for easy/normal regimes, or changing the reward from raw
  acceptance toward throughput-aware utility.
- Rework `EpistemicConsensusK` before using it in conclusions. The current
  log-prob variance penalty and continuation threshold make it collapse to
  one-token drafts.

## Open Questions

- Which stochastic perturbation gives the best signal-to-cost ratio: sampling
  temperature, logit noise, dropout, or a mixture?
- How small can `M` be before consensus becomes too noisy?
- Does consensus predict target acceptance when the drafter is confidently
  wrong?
- Which reward best updates `k`: acceptance rate, accepted tokens, tokens per
  second, or a weighted utility?
- Can the regime model outperform a simpler threshold controller without
  becoming too sensitive to false change points?

## Current Status

- Research direction defined.
- `EpistemicConsensusK` implemented in
  `research/v.poponnikov/experiments/stochastic_dynamic_k.py`.
- `LatentRegimeK` implemented in
  `research/v.poponnikov/experiments/stochastic_dynamic_k.py`.
- Research experiments registered for auto-discovery:
  `stochastic_consensus_k` and `latent_regime_k`.
- Unit tests added in `tests/unit/test_v_poponnikov_dynamic_k.py`.
