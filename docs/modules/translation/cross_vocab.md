# CrossVocabTranslator

Maps probabilities from drafter vocabulary to target vocabulary.

## Rules

- **Rule 1** — exact token string match
- **Rule 2** — approximate redistribution (substring matching)
- **Rule 3 (Lattice)** — exact DAG-based probability computation (optional, replaces Rule 2)
- **Learned Translator** — neural model for unmatched tokens (optional)
