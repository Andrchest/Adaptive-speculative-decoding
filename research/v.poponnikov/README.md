# v.poponnikov Research Area

## Research Direction

This research track studies stochastic methods for choosing the speculative
draft length `k` dynamically during generation. The active method is now
`LatentRegimeK`, an online hidden-regime controller that increases `k` when
drafting looks reliable and shrinks `k` when long drafts are likely to be
rejected by the target model.

The earlier consensus-based method was removed from the active implementation
after smoke and real-model tests showed that it was not viable: it achieved high
acceptance mostly by collapsing to one-token drafts while paying the cost of
multiple drafter trajectories per decision.

## Motivation

Speculative decoding trades off drafter compute against target verification
efficiency. A fixed draft length can be suboptimal because the best `k` changes
across prompts, token positions, and local text regimes. Small `k` limits
speedup, while large `k` can waste drafter work when most draft tokens are
rejected.

This track treats `k` selection as an online stochastic control problem. Each
generation step observes local signals such as acceptance rate, rejection
position, drafter uncertainty, and token class, then samples the next `k` from
an adaptive distribution.

## Active Method: Latent Regime K

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

## Retired Method: Epistemic Consensus K

`EpistemicConsensusK` was tested as a stochastic self-consensus controller. It
sampled several drafter trajectories and used agreement between them as an
uncertainty signal for selecting `k`.

The method was removed from active experiments because it was too expensive and
too conservative:

- Smoke run: 1.93 tokens/sec, mean draft length 1.17, 26.91 s wall time.
- Real Qwen run: 0.72 tokens/sec, mean draft length 1.03, 3691.99 s wall time.
- Real Qwen run selected `k = 1` for 3708 of 3774 selections.

This made the method useful as a negative result, but not as a comparison that
should remain in the active notebook or experiment registry.

## Implementation Plan

1. Implement `LatentRegimeK` as a research adaptive controller. Done.
2. Add a research experiment under `research/v.poponnikov/experiments/`. Done.
3. Compare against `01_baseline` and `08_+speedup_adapt`. Done.
4. Add unit tests for sampling, posterior updates, metric export, and bounds on
   `k`. Done.
5. Add lightweight smoke experiments with tiny models. Done.
6. Add notebook workflow for online IDEs without terminal access. Done.
7. Add required 70M/125M drafter model-matrix benchmark workflow. Done.
8. Tune the regime controller so unavailable uncertainty signals do not force
   hard/transition regimes and successful drafts grow `k` more readily. Done.
9. Run the updated model matrix and compare the less-conservative controller
   against the previous results. Next.

## Metrics

Primary metrics:

- `tokens_per_sec`
- `acceptance_rate`
- `avg_accepted_tokens`
- `avg_draft_length`
- `wall_time_total_s`

Regime-specific metrics:

- mean selected `k`
- distribution of selected `k`
- regime posterior entropy
- change-point reset frequency
- final regime lambdas
- regime distribution

## Experiment Commands

Fast iteration:

```bash
python src/main.py --research --tiny -n 5 --max-new-tokens 32 --no-mlflow
```

Single experiment:

```bash
python src/main.py --experiment latent_regime_k --tiny -n 5 --max-new-tokens 32 --no-mlflow
```

Reference baselines:

```bash
python src/main.py --experiment 01_baseline --tiny -n 5 --max-new-tokens 32 --no-mlflow
python src/main.py --experiment 08_+speedup_adapt --tiny -n 5 --max-new-tokens 32 --no-mlflow
```

## Comparison Workflow

Notebook workflow for online IDEs without terminal access:

1. Open `research/v.poponnikov/notebooks/dynamic_k_comparison.ipynb`.
2. Run the cells from top to bottom.
3. On a fresh Python 3.10 online image, run the dependency install cell once,
   then restart the notebook kernel and set `INSTALL_DEPENDENCIES = False`.
4. Keep the tiny sanity run enabled first.
5. Run the model matrix cell for the required 70M and 125M drafter
   comparisons.

The required matrix is:

| Drafter | Targets |
| --- | --- |
| `EleutherAI/pythia-70m` | `Qwen/Qwen2.5-1.5B-Instruct`, `Qwen/Qwen2.5-3B-Instruct`, `Qwen/Qwen2.5-7B-Instruct` |
| `facebook/opt-125m` | `Qwen/Qwen2.5-1.5B-Instruct`, `Qwen/Qwen2.5-3B-Instruct`, `Qwen/Qwen2.5-7B-Instruct` |

Optional targets `Qwen/Qwen2.5-14B-Instruct` and
`Qwen/Qwen2.5-32B-Instruct` can be enabled in the notebook with
`INCLUDE_LARGE_TARGETS = True` if there is enough GPU memory and time.

Command-line workflow:

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe research\v.poponnikov\notebooks\dynamic_k_comparison.py `
  --matrix `
  --output-dir research\v.poponnikov\results\model_matrix `
  --plots-dir research\v.poponnikov\plots `
  --samples 50 `
  --max-new-tokens 128 `
  --device cuda `
  --draft-sizes 70m 125m `
  --target-sizes 1.5b 3b 7b
```

This runs `01_baseline`, `08_+speedup_adapt`, and `latent_regime_k` in one
comparison pass for every selected drafter-target pair.

Per-pair outputs:

- `research/v.poponnikov/results/model_matrix/70m-1_5b/metrics.csv`
- `research/v.poponnikov/plots/70m-1_5b-plots/comparison.png`

The same pattern is used for `70m-3b`, `70m-7b`, `125m-1_5b`,
`125m-3b`, and `125m-7b`. The aggregate matrix CSV is written to
`research/v.poponnikov/results/model_matrix/model_matrix_metrics.csv`.

To regenerate CSVs and plots from existing JSON results without rerunning
models:

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe research\v.poponnikov\notebooks\dynamic_k_comparison.py `
  --matrix `
  --plot-only `
  --output-dir research\v.poponnikov\results\model_matrix `
  --plots-dir research\v.poponnikov\plots
```

## Preliminary Results

Smoke comparison on `gsm8k`, 5 samples, 32 max new tokens, with
`facebook/opt-125m` as drafter and `facebook/opt-350m` as target:

| Experiment | Tokens/sec | Acceptance rate | Avg accepted | Avg draft length | Wall time |
| --- | ---: | ---: | ---: | ---: | ---: |
| `01_baseline` | 6.93 | 21.29% | 1.09 | 5.00 | 10.24 s |
| `08_+speedup_adapt` | 7.73 | 23.00% | 1.21 | 5.47 | 11.89 s |
| `latent_regime_k` | 10.36 | 54.97% | 1.78 | 3.51 | 10.81 s |

Smoke interpretation:

- `LatentRegimeK` is the strongest smoke result. It improves throughput over
  the fixed baseline and speedup-adaptive baseline while also raising
  acceptance rate.
- The smoke run is too small to prove the regime method is generally better.
  It mainly confirms that the implementation works and that the controller can
  use a nontrivial range of `k` values.

Real-model comparison on `gsm8k`, 50 samples, 128 max new tokens, with
`Qwen/Qwen2.5-0.5B-Instruct` as drafter and
`Qwen/Qwen2.5-7B-Instruct` as target:

| Experiment | Tokens/sec | Acceptance rate | Avg accepted | Avg draft length | Wall time |
| --- | ---: | ---: | ---: | ---: | ---: |
| `01_baseline` | 6.21 | 38.88% | 1.86 | 5.00 | 680.84 s |
| `08_+speedup_adapt` | 5.04 | 40.24% | 1.60 | 4.22 | 797.10 s |
| `latent_regime_k` | 4.52 | 51.41% | 1.16 | 2.33 | 766.38 s |

Real-run interpretation:

- The fixed baseline is the strongest throughput result in the Qwen run. It
  reaches 6.21 tokens/sec, while `latent_regime_k` reaches 4.52 tokens/sec.
- `LatentRegimeK` is still promising as a control signal. It raises acceptance
  from 38.88% to 51.41%, but it reduces mean selected `k` from the fixed value
  of 5.00 to 2.33.
- The result suggests that the posterior is detecting harder regions, but the
  lambda and reward update currently shrink drafts too much.
- Higher acceptance alone is not sufficient; the chosen `k` must be large
  enough to amortize target verification and controller overhead.

## Next Tuning Direction

- Re-run the 70M/125M model matrix after the less-conservative update.
- Compare whether the higher easy/normal floors improve throughput without
  losing too much acceptance.
- Continue changing the reward from raw acceptance toward throughput-aware
  utility if `k` is still too small.
- Run larger comparisons with multiple seeds and confidence intervals.

## Open Questions

- Which reward best updates `k`: acceptance rate, accepted tokens, tokens per
  second, or a weighted utility?
- Can the regime model outperform a simpler threshold controller without
  becoming too sensitive to false change points?
- How much posterior entropy is useful before the controller becomes unstable?
- Should easy and normal regimes have a larger minimum lambda than hard and
  transition regimes?

## Current Status

- Research direction narrowed to `LatentRegimeK`.
- `EpistemicConsensusK` retired and removed from active code because it was too
  slow and collapsed to one-token drafts.
- `LatentRegimeK` implemented in
  `research/v.poponnikov/experiments/stochastic_dynamic_k.py`.
- Only `latent_regime_k` is registered for auto-discovery from this research
  module.
- Notebook comparison now runs `01_baseline`, `08_+speedup_adapt`, and
  `latent_regime_k` across the required 70M/125M drafter matrix.
- Each drafter-target pair writes a per-pair `metrics.csv` and one combined
  `comparison.png`.
- `LatentRegimeK` was tuned to be less conservative: unavailable entropy and
  token-class signals now use neutral defaults, easy/normal regimes have higher
  lambda floors, and successful full drafts grow lambda faster than failed
  drafts shrink it.
- Unit tests updated in `tests/unit/test_v_poponnikov_dynamic_k.py`.
