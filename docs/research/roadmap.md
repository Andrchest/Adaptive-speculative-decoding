# Research Roadmap

## Overview

Research happens independently on per-team-member branches.
Each week, team members sync with `main`, work on their branch,
and present results at the end of the week.

## Current Cycle (Week 1-5)

| Week | m.krylov | v.poponnikov | a.polevoi | al.khadeeva | da.popov | e.pestrovskii |
| (Михаил Крылов) | (Вадим Попонников) | (Андрей Полевой) | (Алия Хадеева) | (Данил Попов) | (Евгений Пестровский) |
|------|----------|--------------|-----------|-------------|----------|---------------|
| 1 | 📖 Read papers | 📖 Read papers | 📖 Read papers | 📖 Read papers | 📖 Read papers | 📖 Read papers |
| 2 | Hypothesis | Hypothesis | Hypothesis | Hypothesis | Hypothesis | Hypothesis |
| 3 | Experiment 1 | Experiment 1 | Experiment 1 | Experiment 1 | Experiment 1 | Experiment 1 |
| 4 | Experiment 2 | Experiment 2 | Experiment 2 | Experiment 2 | Experiment 2 | Experiment 2 |
| 5 | Results | Results | Results | Results | Results | Results |

## Suggested Research Directions

- **Cache**: Better eviction strategies, adaptive cache size
- **Translation**: Better cross-vocab alignment, neural translator
- **Distillation**: Faster convergence, multi-step distillation
- **Adaptive**: Better k-selection, context-aware drafting
- **Routing**: Multi-drafter efficiency, cold-start problem
- **Evaluation**: Better metrics, stress tests, edge cases

## How to Sync

```bash
# Each Monday morning
git checkout main
git pull origin main
git checkout research/<your-name>
git merge main
```
