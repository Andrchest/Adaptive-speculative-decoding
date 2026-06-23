#!/usr/bin/env python3
"""
CLI entry point for Adaptive Speculative Decoding experiments.

Examples:

    # Run full ablation suite
    python src/main.py --suite ablation

    # Run a single named experiment
    python src/main.py --experiment 04_+online_distil

    # Run all research experiments
    python src/main.py --research

    # List research experiments only
    python src/main.py --list --research

    # Quick smoke test (1 sample, tiny models)
    python src/main.py --smoke

    # List available experiments
    python src/main.py --list

    # Run with tiny models for fast testing
    python src/main.py --suite ablation --tiny -n 1
"""

import logging
import sys
from enum import Enum
from typing import Annotated, Literal

import typer
from rich.console import Console
from rich.logging import RichHandler


class LogLevel(str, Enum):
    QUIET = "QUIET"      # tqdm progress only + summary at end
    NORMAL = "NORMAL"    # + warning/error
    VERBOSE = "VERBOSE"  # + all info (legacy behavior)

    def to_logging_level(self) -> int:
        mapping = {"QUIET": logging.WARNING, "NORMAL": logging.WARNING, "VERBOSE": logging.DEBUG}
        return mapping[self.value]


# --- Global logging setup ---
_log_level: LogLevel = LogLevel.QUIET

class _SourceFilter(logging.Filter):
    """Allow only our own source loggers through."""
    _ALLOWED = {"src"}
    def filter(self, record: logging.LogRecord) -> bool:
        if not self._ALLOWED:
            return True
        name = record.name
        return any(n == name or name.startswith(n + ".") for n in self._ALLOWED)


def _setup_logging(level: LogLevel) -> None:
    global _log_level
    _log_level = level

    level_num = level.to_logging_level()
    logging.basicConfig(
        level=level_num,
        format="%(message)s",
        datefmt="[%H:%M:%S]",
        handlers=[RichHandler(rich_tracebacks=True, show_path=True)],
    )
    for noisy in ("urllib3", "httpx", "requests", "transformers", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.CRITICAL)
    # Restrict to our own modules
    logging.getLogger("src").addFilter(_SourceFilter())


# --- Progress helpers (for QUIET/NORMAL mode) ---

import os as _os

def _has_tty() -> bool:
    return _os.isatty(1)


def _init_progress(total: int, desc: str = "") -> object:
    """Return a tqdm progress object or a no-op when stderr is not a tty."""
    try:
        from tqdm import tqdm as _tqdm
        if _has_tty():
            return _tqdm(total=total, desc=desc or None, leave=False, ncols=80)
    except ImportError:
        pass
    # Fallback: no-op
    class _NoOp:
        def __init__(self, *a, **k):
            pass
        def update(self, n=1): pass
        def close(self): pass
    return _NoOp()

for noisy in ("urllib3", "httpx", "requests", "transformers", "huggingface_hub"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

from experiments import (  # noqa: E402
    ABLATION_SUITE,
    BaseExperiment,
    ExperimentRunner,
    discover_experiments,
    discover_research_experiments,
)

logger = logging.getLogger(__name__)
console = Console()


class _SmokeTestExperiment(BaseExperiment):
    """Minimal smoke test: baseline config with tiny models and 1 sample."""

    def __init__(self) -> None:
        from experiments.base import ExperimentMeta
        from experiments.runner import ExperimentConfig

        super().__init__(
            ExperimentMeta(
                name="smoke_test",
                description="Smoke test (1 sample, tiny models)",
                tags=["smoke"],
            )
        )
        self._config_class = ExperimentConfig

    def get_config(self):
        return self._config_class(
            name="smoke_test",
            drafter_model_path="facebook/opt-125m",
            target_model_path="facebook/opt-350m",
            dataset="gsm8k",
            max_samples=1,
            max_new_tokens=32,
            mlflow_experiment="",
        )


def _apply_overrides(
    experiments: list[BaseExperiment],
    *,
    tiny_models: bool = False,
    max_samples: int = 0,
    max_new_tokens: int = 0,
    no_mlflow: bool = False,
) -> None:
    """Apply CLI overrides to experiment configs in-place."""
    for exp in experiments:
        if tiny_models:
            exp.set_config_override("drafter_model_path", "facebook/opt-125m")
            exp.set_config_override("target_model_path", "facebook/opt-350m")
            exp.set_config_override("max_new_tokens", 32)
        if max_samples > 0:
            exp.set_config_override("max_samples", max_samples)
        if max_new_tokens > 0:
            exp.set_config_override("max_new_tokens", max_new_tokens)
        if no_mlflow:
            exp.set_config_override("mlflow_experiment", "")


def main(  # noqa: C901
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
    log_level: Annotated[
        LogLevel,
        typer.Option("--log-level", help="Verbosity: QUIET (tqdm), NORMAL, or VERBOSE (all logs)"),
    ] = LogLevel.QUIET,
    no_mlflow: Annotated[
        bool,
        typer.Option("--no-mlflow", help="Disable MLflow logging"),
    ] = False,
    max_samples: Annotated[
        int | None,
        typer.Option(
            "--max-samples", "-n", help="Override max_samples per experiment (0 = keep default)"
        ),
    ] = 0,
    max_new_tokens: Annotated[
        int | None,
        typer.Option(
            "--max-new-tokens", help="Override max_new_tokens per experiment (0 = keep default)"
        ),
    ] = 0,
    tiny_models: Annotated[
        bool,
        typer.Option("--tiny", "-t", help="Use tiny models (opt-125m/opt-350m) for fast testing"),
    ] = False,
    research: Annotated[
        bool,
        typer.Option(
            "--research",
            "-r",
            help="Run only research experiments (from research/*/experiments/)",
        ),
    ] = False,
) -> None:
    """Adaptive Speculative Decoding — experiment runner."""
    # --- Setup logging ---
    _setup_logging(log_level)

    logger = logging.getLogger(__name__)

    # --- Set global log level BEFORE any experiment runs ---
    import experiments.runner as _rl_mod
    _rl_mod._log_level = log_level.value
    logger.info("Global _log_level set to: %s", log_level.value)
    logger.info(
        "Parsed arguments: suite=%s experiment=%s smoke=%s list=%s max_samples=%d max_new_tokens=%d log_level=%s",
        suite,
        experiment,
        smoke,
        list_experiments,
        max_samples or 0,
        max_new_tokens or 0,
        log_level.value,
    )

    # --- List experiments ---
    if list_experiments:
        logger.info("Listing available experiments")
        if research:
            all_exps = discover_research_experiments()
            console.print("[bold]Research experiments:[/bold]\n")
        else:
            all_exps = discover_experiments()
            console.print("[bold]Available experiments:[/bold]\n")
        for exp in all_exps:
            tags = f" [{', '.join(exp.meta.tags)}]" if exp.meta.tags else ""
            desc = f" — {exp.meta.description}" if exp.meta.description else ""
            console.print(f"  [cyan]{exp.meta.name}[/]{tags}{desc}")
        if not all_exps:
            if research:
                console.print(
                    "  [dim](no research experiments found — "
                    "create research/<name>/experiments/*.py)[/dim]"
                )
            else:
                console.print("  [dim](no experiments found)[/dim]")
        return

    # --- Smoke test ---
    if smoke:
        logger.info("Starting smoke test run")
        experiments = [_SmokeTestExperiment()]
        runner = ExperimentRunner(experiments=experiments, output_dir=output_dir, device=device)
        logger.info("Running smoke test")
        results = runner.run_all()
        logger.info("Smoke test run finished")
        _print_summary(results)
        return

    # --- Select experiments ---
    if research:
        logger.info("Discovering research experiments")
        experiments = discover_research_experiments()
        if not experiments:
            logger.error("No research experiments found")
            console.print(
                "[red]No research experiments found.[/red]\n"
                "[dim]Create research/<name>/experiments/<file>.py and add __all__ = [YourClass][/dim]"
            )
            sys.exit(1)
        logger.info("Found %d research experiment(s)", len(experiments))
    elif suite == "ablation":
        logger.info("Selected ablation suite with %d experiments", len(ABLATION_SUITE))
        experiments = list(ABLATION_SUITE)
    elif suite == "cache":
        logger.info("Selected cache suite")
        from experiments.suites import CACHE_SUITE

        experiments = list(CACHE_SUITE)
    elif suite == "dataset":
        logger.info("Selected dataset suite")
        from experiments.suites import DATASET_SUITE

        experiments = list(DATASET_SUITE)
    elif experiment:
        logger.info("Selected single experiment: %s", experiment)
        all_exps = discover_experiments()
        matching = [e for e in all_exps if e.meta.name == experiment]
        if not matching:
            logger.error("No experiment named %r", experiment)
            console.print(f"[red]No experiment named {experiment!r}[/red]")
            sys.exit(1)
        experiments = matching
    else:
        logger.error("No experiment selection provided")
        console.print(
            "[red]Specify --suite, --experiment, --research, or --smoke. Use --list to see options.[/red]"
        )
        sys.exit(1)

    # --- Apply CLI overrides ---
    _apply_overrides(
        experiments,
        tiny_models=tiny_models,
        max_samples=max_samples or 0,
        max_new_tokens=max_new_tokens or 0,
        no_mlflow=no_mlflow,
    )
    # Set log_level on each experiment config
    for exp in experiments:
        exp.set_config_override("log_level", log_level.value)

    # --- Run ---
    runner = ExperimentRunner(experiments=experiments, output_dir=output_dir, device=device)
    logger.info("Running %d experiment(s)", len(experiments))
    results = runner.run_all()
    logger.info("All experiments finished")

    _print_summary(results)


def _print_summary(results: list[dict]) -> None:
    """Print a comprehensive comparison table to the console."""
    if not results:
        return

    console.print("\n")
    console.print("[bold]" + "=" * 70 + "[/]")
    console.print("[bold]  Final Comparison[/]")
    console.print("[bold]" + "=" * 70 + "[/]")

    # Sort by wall_time_total_s (fastest first)
    sorted_results = sorted(
        results,
        key=lambda r: r["metrics"].get("wall_time_total_s", float("inf"))
    )

    # Find fastest and slowest for badges
    fastest = sorted_results[0]
    slowest = sorted_results[-1]
    fastest_tps = max(r["metrics"].get("tokens_per_sec", 0) for r in sorted_results)

    # Header
    console.print(
        "  " + f'{"#":<4}'
        + f'{"Experiment":<24} '
        + f'{"Acc":>7} '
        + f'{"TPS":>7} '
        + f'{"Wall(s)":>9} '
        + f'{"Acc/Avg":>10} '
        + f'{"GPU(GB)":>10} '
        + f'{"Status":<10}'
    )
    console.print("[dim]" + "─" * 90 + "[/]")

    for rank, r in enumerate(sorted_results, 1):
        m = r["metrics"]
        name = r["config"]["name"]
        acc = m.get("acceptance_rate", 0)
        tps = m.get("tokens_per_sec", 0)
        wall = m.get("wall_time_total_s", 0)
        avg_acc = m.get("avg_accepted_tokens", 0)
        gpu = m.get("gpu_mem_peak_gb", 0)

        # Determine badge
        badge = ""
        if r is fastest:
            badge = "⚡ Fastest"
        elif r is slowest:
            badge = "🐌 Slowest"

        console.print(
            f"  {rank:<4}"
            f"[cyan]{name}[/]" if len(name) <= 24 else f"  {rank:<4}[cyan]{name[:24]}[/]",
            f"{acc:>6.1f}%",
            f"{tps:>7.1f}",
            f"{wall:>9.3f}",
            f"{avg_acc:>5.2f}/5.0",
            f"{gpu:>10.2f}",
            f"{badge:<10}",
        )

    console.print("[bold]" + "=" * 70 + "[/]")

    # Results file paths
    result_files = []
    for r in sorted_results:
        name = r["config"]["name"]
        result_files.append(f"results/{name}.json")

    console.print(f"  Results: {', '.join(result_files[:5])}")
    if len(result_files) > 5:
        console.print(f"  ... and {len(result_files) - 5} more")
    console.print(f"  CSV:     results/comparison_table.csv")
    console.print("[bold]" + "=" * 70 + "[/]")


if __name__ == "__main__":
    typer.run(main)
