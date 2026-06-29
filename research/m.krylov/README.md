# m.krylov — Research Area

## Name
Mikhail Krylov

## Hypothesis / Research Direction

Bandit-based routing for speculative decoding with online distillation.

Instead of using a static BERT-based classifier to assign draft models by prompt domain (as in *Online Speculative Decoding*), use a multi-armed bandit algorithm that learns routing policy online from actual decoding performance. The router selects a drafter, observes a reward based on acceptance rate and wall-clock time, and updates its policy — all while the drafters are simultaneously improved through online distillation.

### Data Flow

1. A prompt arrives.
2. A bandit-based router selects a draft model αᵢ.
3. The selected model generates K tokens for the current prefix.
4. The target model verifies the draft tokens.
5. A subset of draft tokens is accepted. Target tokens that correct draft errors are appended.
6. A reward rᵢ is computed.
7. The router updates its bandit policy based on the reward.
8. The accepted prefix and target logits are stored in a distillation buffer.
9. Periodically, the draft models are updated using data from the buffer (KL divergence + N-gram NLL loss).

### Reward Definition

At each speculation step:

- **A**: Number of accepted draft tokens
- **T_draft(a)**: Time to generate the draft tokens (forward pass + sampling)
- **T_target**: Time for the target model to verify the tokens
- **Reward**: rᵢ = A / (T_draft(a) + T_target)

This reward balances acceptance quality against wall-clock throughput — a drafter that is fast but inaccurate, or accurate but slow, will both be penalized.

### Candidate Algorithms

1. **Upper Confidence Bound (UCB)** — simple, no hyperparameters beyond exploration coefficient
2. **Thompson Sampling** — Bayesian approach, naturally balances exploration/exploitation
3. **Contextual Bandit** — uses prompt features (length, domain, token distribution) as context

### Novelty Assessment

The closest existing work is *Online Speculative Decoding*, which combines online distillation with model routing. Their routing uses a BERT-based classifier that assigns draft models by prompt domain. Since they do not employ a bandit-based approach, our method introduces a clear novelty gap: the routing policy learns directly from decoding performance (acceptance rate + wall-clock time) rather than from static prompt features.

## Tasks

### Phase 1 — UCB only, reward signal (no distillation)
- [x] Implement UCB1 router
- [x] Reward computation: `r = accepted / (T_draft + T_target)`
- [x] Verify reward values are reasonable (tokens/sec range) — *fixed bug: hook now receives real StepResult data*
- [x] Verify bandit updates (pulls, means) are logged correctly — *fixed bug: rewards are now non-zero*

### Phase 2 — Enable arm switching
- [x] Thompson Sampling with Normal-Gamma posterior
- [x] Observe exploration → convergence in logs — *arm selections logged at DEBUG level; check `bandit_selections` in extra_state*
- [x] Verify active drafter changes during decoding — *DualRouter tracks agreements/disagreements vs MLP*
- [ ] Compare UCB vs Thompson on same dataset — *run `bandit_ucb` and `bandit_thompson` with same seed, compare `bandit_mean_reward` and arm distributions*

### Phase 2b — Non-stationary adaptation (new)
- [x] Sliding reward window (`reward_window` param on `DrafterEntry`) so bandit adapts when distillation shifts arm quality
- [x] Per-step bandit updates (`per_step_update` flag) for finer-grained learning signals
- [x] Seeded RNG in `PerArmBuffer` for reproducible sampling

### Phase 3 — Add per-arm distillation
- [x] Per-arm distillation buffer (tagged with arm index)
- [x] Periodic replay: each drafter trained only on its own data — *`_replay_for_arm` filters by arm_idx, called every `replay_every` prompts*
- [ ] Monitor how drafter updates affect bandit behaviour — *use `reward_window > 0` to track adaptation*
- [ ] Tune replay frequency and batch size

### Phase 4 — Full comparison
- [x] Compare against `09_+routing` (MLP-based router) — *`BanditVsMLPExperiment` with DualRouter, MLP trained online*
- [x] Evaluate on multiple datasets (gsm8k, mbpp, alpaca, xsum) — *`BanditMultiDatasetExperiment`*
- [ ] Analyse exploration vs exploitation trade-off — *run with varying `exploration` c values*
- [x] Contextual bandit with prompt features — *`BanditContextualExperiment` (LinUCB, 8 features, vocab_size-aware)*

## Implementation

See `experiments/bandit_routing.py` for the full implementation.

Key classes:
- `UCBBanditRouter` — UCB1 with exploration coefficient `c`
- `ThompsonSamplingRouter` — Normal-Gamma posterior for continuous rewards
- `PerArmBuffer` — FIFO buffer tagged by arm index for per-drafter distillation
- `BanditRoutingExperiment` — main experiment class with phased enablement

## References

- Online Speculative Decoding (closest prior work — BERT-based routing + distillation)
- UCB1: Auer et al. 2002 "Finite-time Analysis of the Multiarmed Bandit Problem"
- Thompson Sampling: Thompson 1933 "On the likelihood that one unknown probability exceeds another"
- Normal-Gamma conjugate prior: Gelman et al. "Bayesian Data Analysis" Ch. 3

## References

- Online Speculative Decoding (closest prior work — BERT-based routing + distillation)
- Multi-armed bandit literature: UCB (Auer et al. 2002), Thompson Sampling (Thompson 1933)
- Contextual bandits: LinUCB, neural bandits

## Notes

Discussed with DeepSeek to refine the hypothesis and data flow.
