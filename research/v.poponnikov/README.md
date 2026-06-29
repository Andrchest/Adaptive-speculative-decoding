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
