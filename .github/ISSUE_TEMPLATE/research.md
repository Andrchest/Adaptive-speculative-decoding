---
name: Research Task
about: Plan a research investigation
title: "research: "
labels: research
---

## Researcher
<!-- Who is working on this? -->

## Hypothesis
<!-- What do you believe will happen? -->

## Background
<!-- Prior work, papers, related experiments -->

## Experiments
<!-- What experiments will you run? -->
1. 
2. 

### How to implement

The project uses a **Strategy-pattern experiment framework**. Each experiment
is a `BaseExperiment` subclass in `research/<username>/experiments/`:

```bash
mkdir -p research/<your_name>/experiments
cp src/experiments/templates/minimal_template.py \
   research/<your_name>/experiments/<experiment_name>.py
# Edit: override get_config() and optional build_* / on_* methods
# Register class in __all__ at the bottom
```

Run experiments: `python src/main.py --research` or `--experiment <name>`.

See `src/experiments/templates/minimal_template.py` and `research/README.md`.

## Success Criteria
<!-- How will you know if it worked? -->
- [ ] Metric 1 improves by X%
- [ ] Metric 2 stays within Y%
- [ ] Ablation shows component Z contributes

## Results
<!-- Fill after experiment -->
- [ ] Acceptance rate:
- [ ] Speedup:
- [ ] GPU memory:

## Conclusions
<!-- What did you learn? What's next? -->
