# Research Area — Shared

Each team member has their own folder here for storing:
- Experimental notebooks (`.ipynb`)
- Results (`.csv`, `.json`)
- Configuration files (`.yaml`)
- Notes (`README.md`)

## Research Folder Structure

```
research/<username>/
├── README.md              — project description, hypotheses, tasks
│   └── configs/             — experiment configurations
│       └── *.yaml
├── results/               — results
│   ├── *.csv
│   └── *.json
├── notebooks/             — Jupyter notebooks
│   └── *.ipynb
└── plots/                 — plots
    └── *.png
```

## Rules

- Each research branch is created from `main` and lives independently
- Results are written to `research/<username>/results/`
- Coding experiments go in `src/`, research analysis goes in notebooks
- Before merging to `main` — at least one passing test
