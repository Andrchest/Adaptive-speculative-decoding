"""Run and plot comparisons for v.poponnikov stochastic dynamic-k experiments.

The experiment logic stays in ``research/v.poponnikov/experiments``.  This file
is the research analysis layer: it runs the relevant baselines and dynamic-k
experiments, writes merged metrics, and produces plots for the research folder.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")

from experiments import ExperimentRunner, discover_experiments

if TYPE_CHECKING:
    from collections.abc import Sequence

DEFAULT_EXPERIMENTS = (
    "01_baseline",
    "08_+speedup_adapt",
    "stochastic_consensus_k",
    "latent_regime_k",
)

PRIMARY_METRICS = (
    "tokens_per_sec",
    "acceptance_rate",
    "avg_accepted_tokens",
    "avg_draft_length",
    "wall_time_total_s",
)


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the comparison run."""
    parser = argparse.ArgumentParser(
        description="Run stochastic dynamic-k comparisons and generate plots.",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=list(DEFAULT_EXPERIMENTS),
        help="Experiment names to compare, in order.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("research/v.poponnikov/results/dynamic_k_comparison"),
        help="Directory for JSON and CSV results.",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=Path("research/v.poponnikov/plots/dynamic_k_comparison"),
        help="Directory for generated plots.",
    )
    parser.add_argument("--device", default="cuda", help="Torch device, usually cuda or cpu.")
    parser.add_argument("--samples", type=int, default=5, help="Number of dataset samples.")
    parser.add_argument("--max-new-tokens", type=int, default=32, help="Generation budget.")
    parser.add_argument(
        "--tiny",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use opt-125m/opt-350m for fast iteration.",
    )
    parser.add_argument("--drafter-model", default="", help="Override drafter model.")
    parser.add_argument("--target-model", default="", help="Override target model.")
    parser.add_argument(
        "--enable-mlflow",
        action="store_true",
        help="Keep MLflow enabled. By default this script disables MLflow.",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Read existing result JSON files and regenerate CSV/plots.",
    )
    return parser.parse_args()


def select_experiments(names: Sequence[str]) -> list[object]:
    """Discover and return experiments by name, preserving requested order.

    Args:
        names: Experiment identifiers such as ``01_baseline``.

    Returns:
        Instantiated experiment objects.

    Raises:
        ValueError: If any requested experiment is not discoverable.
    """
    available = {experiment.meta.name: experiment for experiment in discover_experiments()}
    missing = [name for name in names if name not in available]
    if missing:
        known = ", ".join(sorted(available))
        raise ValueError(f"Unknown experiment(s): {missing}. Known experiments: {known}")
    return [available[name] for name in names]


def apply_common_overrides(
    experiments: Sequence[object],
    *,
    tiny: bool,
    samples: int,
    max_new_tokens: int,
    device: str,
    drafter_model: str = "",
    target_model: str = "",
    enable_mlflow: bool = False,
) -> None:
    """Apply the same runtime settings to every selected experiment."""
    for experiment in experiments:
        if tiny:
            experiment.set_config_override("drafter_model_path", "facebook/opt-125m")
            experiment.set_config_override("target_model_path", "facebook/opt-350m")
        if drafter_model:
            experiment.set_config_override("drafter_model_path", drafter_model)
        if target_model:
            experiment.set_config_override("target_model_path", target_model)
        if samples > 0:
            experiment.set_config_override("max_samples", samples)
        if max_new_tokens > 0:
            experiment.set_config_override("max_new_tokens", max_new_tokens)
        if not enable_mlflow:
            experiment.set_config_override("mlflow_experiment", "")
        if device == "cpu":
            experiment.set_config_override("target_use_4bit", False)
        experiment.set_config_override("log_level", "QUIET")


def run_comparison(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Run the requested experiments through the shared project runner."""
    experiments = select_experiments(args.experiments)
    apply_common_overrides(
        experiments,
        tiny=args.tiny,
        samples=args.samples,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
        drafter_model=args.drafter_model,
        target_model=args.target_model,
        enable_mlflow=args.enable_mlflow,
    )
    runner = ExperimentRunner(
        experiments=experiments,
        output_dir=str(args.output_dir),
        device=args.device,
    )
    return runner.run_all()


def failed_experiments(results: Sequence[dict[str, Any]]) -> list[str]:
    """Return experiment names that failed inside the shared runner."""
    failed: list[str] = []
    for result in results:
        metrics = result.get("metrics", {})
        if metrics.get("error") is True:
            name = str(result.get("config", {}).get("name", metrics.get("name", "unknown")))
            error_type = metrics.get("error_type", "error")
            message = metrics.get("error_message", "")
            failed.append(f"{name}: {error_type}: {message}")
    return failed


def load_results(result_dir: Path, names: Sequence[str]) -> list[dict[str, Any]]:
    """Load per-experiment result JSON files from a result directory."""
    results: list[dict[str, Any]] = []
    missing: list[str] = []
    for name in names:
        path = result_dir / f"{name.replace(' ', '_')}.json"
        if not path.exists():
            missing.append(str(path))
            continue
        with path.open("r", encoding="utf-8") as file:
            results.append(json.load(file))
    if missing:
        raise FileNotFoundError("Missing result JSON file(s): " + ", ".join(missing))
    return results


def metric_keys_union(results: Sequence[dict[str, Any]]) -> list[str]:
    """Return metric keys in first-seen order across all results."""
    keys: list[str] = []
    seen: set[str] = set()
    for result in results:
        for key in result.get("metrics", {}):
            if key not in seen:
                seen.add(key)
                keys.append(key)
    return keys


def write_union_csv(results: Sequence[dict[str, Any]], path: Path) -> None:
    """Write a comparison CSV that includes research-specific metric columns."""
    path.parent.mkdir(parents=True, exist_ok=True)
    metric_keys = metric_keys_union(results)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["experiment", *metric_keys])
        writer.writeheader()
        for result in results:
            metrics = result.get("metrics", {})
            row: dict[str, Any] = {
                "experiment": result.get("config", {}).get("name", metrics.get("name", "")),
            }
            for key in metric_keys:
                row[key] = _csv_value(metrics.get(key, ""))
            writer.writerow(row)


def plot_results(results: Sequence[dict[str, Any]], plots_dir: Path) -> list[Path]:
    """Generate all available comparison plots.

    Args:
        results: Experiment results returned by the runner or loaded from JSON.
        plots_dir: Destination directory.

    Returns:
        Paths of the generated plot files.
    """
    plots_dir.mkdir(parents=True, exist_ok=True)
    plt = _load_matplotlib()
    created = [
        _plot_primary_metrics(results, plots_dir, plt),
        _plot_dynamic_k_summary(results, plots_dir, plt),
    ]
    distribution_plot = _plot_k_distribution(results, plots_dir, plt)
    if distribution_plot is not None:
        created.append(distribution_plot)
    diagnostics_plot = _plot_controller_diagnostics(results, plots_dir, plt)
    if diagnostics_plot is not None:
        created.append(diagnostics_plot)
    return created


def _csv_value(value: Any) -> Any:
    if isinstance(value, dict | list):
        return json.dumps(value, sort_keys=True)
    return value


def _experiment_names(results: Sequence[dict[str, Any]]) -> list[str]:
    return [
        str(result.get("config", {}).get("name", result.get("metrics", {}).get("name", "")))
        for result in results
    ]


def _metric_float(result: dict[str, Any], key: str) -> float | None:
    value = result.get("metrics", {}).get(key)
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


def _load_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "Plotting requires matplotlib. Install the viz dependencies, for example: "
            "uv sync --group viz"
        ) from exc
    return plt


def _plot_primary_metrics(results: Sequence[dict[str, Any]], plots_dir: Path, plt) -> Path:
    names = _experiment_names(results)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes_flat = list(axes.reshape(-1))
    colors = ["#2563eb", "#16a34a", "#dc2626", "#9333ea", "#f59e0b"]

    for axis, metric in zip(axes_flat, PRIMARY_METRICS, strict=False):
        values = [_metric_float(result, metric) or 0.0 for result in results]
        if metric == "acceptance_rate":
            values = [value * 100.0 for value in values]
            ylabel = "percent"
        elif metric == "wall_time_total_s":
            ylabel = "seconds"
        else:
            ylabel = metric
        axis.bar(names, values, color=colors[: len(names)])
        axis.set_title(metric)
        axis.set_ylabel(ylabel)
        axis.tick_params(axis="x", rotation=25)
        for label in axis.get_xticklabels():
            label.set_horizontalalignment("right")

    for axis in axes_flat[len(PRIMARY_METRICS) :]:
        axis.axis("off")

    fig.tight_layout()
    path = plots_dir / "primary_metrics.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _plot_dynamic_k_summary(results: Sequence[dict[str, Any]], plots_dir: Path, plt) -> Path:
    names = _experiment_names(results)
    mean_k = [_dynamic_mean_k(result) for result in results]
    accepted = [_metric_float(result, "avg_accepted_tokens") or 0.0 for result in results]

    x_positions = list(range(len(names)))
    width = 0.38
    fig, axis = plt.subplots(figsize=(11, 5))
    axis.bar(
        [position - width / 2 for position in x_positions],
        mean_k,
        width=width,
        label="mean selected k / draft length",
        color="#2563eb",
    )
    axis.bar(
        [position + width / 2 for position in x_positions],
        accepted,
        width=width,
        label="avg accepted tokens",
        color="#16a34a",
    )
    axis.set_xticks(x_positions, names, rotation=25, ha="right")
    axis.set_ylabel("tokens")
    axis.set_title("Dynamic-k selection versus accepted tokens")
    axis.legend()
    fig.tight_layout()
    path = plots_dir / "dynamic_k_summary.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _plot_k_distribution(results: Sequence[dict[str, Any]], plots_dir: Path, plt) -> Path | None:
    distributions: list[tuple[str, dict[int, float]]] = []
    for result in results:
        metrics = result.get("metrics", {})
        name = str(result.get("config", {}).get("name", metrics.get("name", "")))
        distribution = metrics.get("consensus_k_k_distribution")
        if not isinstance(distribution, dict):
            distribution = metrics.get("regime_k_k_distribution")
        if isinstance(distribution, dict) and distribution:
            distributions.append((name, _normalize_distribution(distribution)))

    if not distributions:
        return None

    all_k = sorted({k for _, distribution in distributions for k in distribution})
    x_positions = list(range(len(all_k)))
    width = min(0.8 / max(len(distributions), 1), 0.35)

    fig, axis = plt.subplots(figsize=(10, 5))
    for index, (name, distribution) in enumerate(distributions):
        offset = (index - (len(distributions) - 1) / 2) * width
        values = [distribution.get(k, 0.0) for k in all_k]
        axis.bar(
            [position + offset for position in x_positions],
            values,
            width=width,
            label=name,
        )

    axis.set_xticks(x_positions, [str(k) for k in all_k])
    axis.set_xlabel("selected k")
    axis.set_ylabel("steps")
    axis.set_title("Selected-k distribution")
    axis.legend()
    fig.tight_layout()
    path = plots_dir / "dynamic_k_distribution.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _plot_controller_diagnostics(
    results: Sequence[dict[str, Any]], plots_dir: Path, plt
) -> Path | None:
    diagnostic_keys = (
        "consensus_k_theta_final",
        "consensus_k_consensus_mean",
        "consensus_k_continue_prob_mean",
        "regime_k_posterior_entropy_mean",
        "regime_k_change_point_mean",
        "regime_k_lambda_easy",
        "regime_k_lambda_normal",
        "regime_k_lambda_hard",
        "regime_k_lambda_transition",
    )
    rows: list[tuple[str, float]] = []
    for result in results:
        name = str(result.get("config", {}).get("name", ""))
        for key in diagnostic_keys:
            value = _metric_float(result, key)
            if value is not None:
                rows.append((f"{name}: {key}", value))

    if not rows:
        return None

    labels = [row[0] for row in rows]
    values = [row[1] for row in rows]
    fig, axis = plt.subplots(figsize=(12, max(5, len(rows) * 0.38)))
    axis.barh(labels, values, color="#0f766e")
    axis.set_title("Dynamic-k controller diagnostics")
    axis.set_xlabel("value")
    fig.tight_layout()
    path = plots_dir / "controller_diagnostics.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _dynamic_mean_k(result: dict[str, Any]) -> float:
    for key in (
        "consensus_k_mean_selected_k",
        "regime_k_mean_selected_k",
        "avg_draft_length",
    ):
        value = _metric_float(result, key)
        if value is not None:
            return value
    return 0.0


def _normalize_distribution(distribution: dict[Any, Any]) -> dict[int, float]:
    normalized: dict[int, float] = {}
    for key, value in distribution.items():
        try:
            normalized[int(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return normalized


def main() -> None:
    """Run or load experiments, then write merged results and plots."""
    args = parse_args()
    if args.plot_only:
        results = load_results(args.output_dir, args.experiments)
    else:
        results = run_comparison(args)

    merged_csv = args.output_dir / "dynamic_k_comparison.csv"
    write_union_csv(results, merged_csv)
    failures = failed_experiments(results)
    if failures:
        print(f"Merged CSV saved to: {merged_csv}")
        raise RuntimeError("Experiment failure(s): " + " | ".join(failures))

    plot_paths = plot_results(results, args.plots_dir)

    print(f"Merged CSV saved to: {merged_csv}")
    for path in plot_paths:
        print(f"Plot saved to: {path}")


if __name__ == "__main__":
    main()
