#!/usr/bin/env python3
"""
CLI entry point for Adaptive Speculative Decoding experiments.

Examples:

    # Run full ablation suite
    python src/main.py --suite ablation

    # Run a single named experiment
    python src/main.py --experiment 04_+online_distil

    # Quick smoke test (1 sample, tiny models)
    python src/main.py --smoke

    # List available experiments
    python src/main.py --list
"""

import logging
import sys
from typing import Annotated, Literal

import typer
from rich.console import Console

from experiments.runner import ABLATION_SUITE, ExperimentConfig, ExperimentRunner

logger = logging.getLogger(__name__)
console = Console()


def main(
    suite: Annotated[
        Literal["ablation", "cache", "dataset"] | None,
        typer.Option("--suite", "-s", help="Run a pre-defined experiment suite"),
    ] = None,
    experiment: Annotated[
        str | None,
        typer.Option("--experiment", "-e", help="Run a single named experiment"),
    ] = None,
    smoke: Annotated[
        bool,
        typer.Option("--smoke", help="Smoke test with 1 sample and smallest models"),
    ] = False,
    list_experiments: Annotated[
        bool,
        typer.Option("--list", "-l", help="List experiments and exit"),
    ] = False,
    output_dir: Annotated[
        str,
        typer.Option("--output-dir", "-o", help="Where to write results"),
    ] = "results",
    device: Annotated[
        str,
        typer.Option("--device", "-d", help="torch device"),
    ] = "cuda",
    no_mlflow: Annotated[
        bool,
        typer.Option("--no-mlflow", help="Disable MLflow logging"),
    ] = False,
    max_samples: Annotated[
        int | None,
        typer.Option("--max-samples", "-n", help="Override max_samples per experiment (0 = keep default)"),
    ] = 0,
    max_new_tokens: Annotated[
        int | None,
        typer.Option("--max-new-tokens", help="Override max_new_tokens per experiment (0 = keep default)"),
    ] = 0,
    tiny_models: Annotated[
        bool,
        typer.Option("--tiny", "-t", help="Use tiny models (opt-125m/opt-350m) for fast testing"),
    ] = False,
) -> None:
    """Adaptive Speculative Decoding — experiment runner."""
    logger.info(
        "Parsed arguments: suite=%s experiment=%s smoke=%s list=%s max_samples=%d max_new_tokens=%d",
        suite,
        experiment,
        smoke,
        list_experiments,
        max_samples,
        max_new_tokens,
    )

    if list_experiments:
        logger.info("Listing available experiments")
        console.print("[bold]Available experiments:[/bold]")
        for cfg in ABLATION_SUITE:
            console.print(f"  {cfg.name}")
        return

    # --- Smoke test ---
    if smoke:
        logger.info("Starting smoke test run")
        configs = [
            ExperimentConfig(
                name="smoke_test",
                drafter_model_path="facebook/opt-125m",
                target_model_path="facebook/opt-350m",
                dataset="gsm8k",
                max_samples=1,
                max_new_tokens=32,
                mlflow_experiment="" if no_mlflow else "adaptive_speculative_smoke",
            )
        ]
        ExperimentRunner(configs, output_dir=output_dir, device=device).run_all()
        logger.info("Smoke test run finished")
        return

    # --- Suite ---
    if suite == "ablation":
        logger.info("Selected ablation suite with %d experiments", len(ABLATION_SUITE))
        configs = ABLATION_SUITE
    elif suite == "cache":
        logger.info("Selected cache suite")
        from experiments.cache_ablation import CACHE_SUITE

        configs = CACHE_SUITE
    elif suite == "dataset":
        logger.info("Selected dataset suite")
        from experiments.dataset_ablation import DATASET_SUITE

        configs = DATASET_SUITE
    elif experiment:
        logger.info("Selected single experiment: %s", experiment)
        matching = [c for c in ABLATION_SUITE if c.name == experiment]
        if not matching:
            logger.error("No experiment named %r", experiment)
            console.print(f"[red]No experiment named {experiment!r}[/red]")
            sys.exit(1)
        configs = matching
    else:
        logger.error("No experiment selection provided")
        console.print(
            "[red]Specify --suite, --experiment, or --smoke. Use --list to see options.[/red]"
        )
        sys.exit(1)

    if no_mlflow:
        logger.info("MLflow logging disabled")
        for cfg in configs:
            cfg.mlflow_experiment = ""

    # --- Tiny models override ---
    if tiny_models:
        logger.info("Overriding to tiny models: opt-125m/opt-350m")
        for cfg in configs:
            cfg.drafter_model_path = "facebook/opt-125m"
            cfg.target_model_path = "facebook/opt-350m"
            cfg.max_new_tokens = min(cfg.max_new_tokens, 32) if cfg.max_new_tokens else 32

    # Apply CLI overrides
    if max_samples is not None and max_samples > 0:
        logger.info("Overriding max_samples to %d", max_samples)
        for cfg in configs:
            cfg.max_samples = max_samples
    if max_new_tokens is not None and max_new_tokens > 0:
        logger.info("Overriding max_new_tokens to %d", max_new_tokens)
        for cfg in configs:
            cfg.max_new_tokens = max_new_tokens

    runner = ExperimentRunner(configs, output_dir=output_dir, device=device)
    logger.info("Running %d experiment(s)", len(configs))
    results = runner.run_all()
    logger.info("All experiments finished")

    console.print("\n[bold]=== Final Comparison ===[/bold]")
    console.print(f"{'Experiment':<30} {'Acc Rate':>10} {'TPS':>10} {'Speedup':>10}")
    console.print("-" * 65)
    for r in results:
        m = r["metrics"]
        console.print(
            f"{r['config']['name']:<30}"
            f"  {m.get('acceptance_rate', 0):.3f}"
            f"  {m.get('tokens_per_sec', 0):8.1f}"
            f"  {m.get('wall_clock_speedup', 0):8.2f}x"
        )


if __name__ == "__main__":
    typer.run(main)
