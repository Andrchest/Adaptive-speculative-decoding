"""Run and plot comparisons for v.poponnikov stochastic dynamic-k experiments.

The experiment logic stays in ``research/v.poponnikov/experiments``.  This file
is the research analysis layer: it runs the relevant baselines and dynamic-k
experiments, writes merged metrics, and produces plots for the research folder.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")

if TYPE_CHECKING:
    from collections.abc import Sequence

DEFAULT_EXPERIMENTS = (
    "01_baseline",
    "08_+speedup_adapt",
    "latent_regime_k",
    "latent_regime_categorical_k",
)

_EXPERIMENT_PROTOTYPES: dict[str, object] | None = None

PRIMARY_METRICS = (
    "tokens_per_sec",
    "acceptance_rate",
    "avg_accepted_tokens",
    "avg_draft_length",
    "wall_time_total_s",
)

DEFAULT_MATRIX_DRAFTS = ("70m", "125m")
DEFAULT_MATRIX_TARGETS = ("1.5b", "3b", "7b")
OPTIONAL_LARGE_TARGETS = ("14b", "32b")


@dataclass(frozen=True)
class ModelSpec:
    """Named model option used by the matrix benchmark."""

    key: str
    label: str
    path: str


@dataclass(frozen=True)
class ModelPair:
    """Drafter-target pair for one comparison run."""

    slug: str
    drafter: ModelSpec
    target: ModelSpec

    @property
    def title(self) -> str:
        """Human-readable plot title."""
        return f"{self.drafter.label} drafter -> {self.target.label} target"


DRAFT_MODELS = {
    "70m": ModelSpec("70m", "70M", "EleutherAI/pythia-70m"),
    "125m": ModelSpec("125m", "125M", "facebook/opt-125m"),
}

TARGET_MODELS = {
    "1.5b": ModelSpec("1.5b", "1.5B", "Qwen/Qwen2.5-1.5B-Instruct"),
    "3b": ModelSpec("3b", "3B", "Qwen/Qwen2.5-3B-Instruct"),
    "7b": ModelSpec("7b", "7B", "Qwen/Qwen2.5-7B-Instruct"),
    "14b": ModelSpec("14b", "14B", "Qwen/Qwen2.5-14B-Instruct"),
    "32b": ModelSpec("32b", "32B", "Qwen/Qwen2.5-32B-Instruct"),
}


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
        "--matrix",
        action="store_true",
        help="Run the configured drafter-target model matrix.",
    )
    parser.add_argument(
        "--draft-sizes",
        nargs="+",
        choices=tuple(DRAFT_MODELS),
        default=list(DEFAULT_MATRIX_DRAFTS),
        help="Draft model sizes to use in --matrix mode.",
    )
    parser.add_argument(
        "--target-sizes",
        nargs="+",
        choices=tuple(TARGET_MODELS),
        default=list(DEFAULT_MATRIX_TARGETS),
        help="Target model sizes to use in --matrix mode.",
    )
    parser.add_argument(
        "--include-large-targets",
        action="store_true",
        help="Also include optional 14B and 32B targets in --matrix mode.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue matrix runs after a failed experiment and still save available outputs.",
    )
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


def build_model_pairs(args: argparse.Namespace) -> list[ModelPair]:
    """Build the requested drafter-target pairs for matrix mode."""
    target_sizes = list(args.target_sizes)
    if args.include_large_targets:
        for target in OPTIONAL_LARGE_TARGETS:
            if target not in target_sizes:
                target_sizes.append(target)

    pairs: list[ModelPair] = []
    for draft_size in args.draft_sizes:
        for target_size in target_sizes:
            drafter = DRAFT_MODELS[draft_size]
            target = TARGET_MODELS[target_size]
            pairs.append(
                ModelPair(
                    slug=f"{_slugify(drafter.key)}-{_slugify(target.key)}",
                    drafter=drafter,
                    target=target,
                )
            )
    return pairs


def select_experiments(names: Sequence[str]) -> list[object]:
    """Discover and return experiments by name, preserving requested order.

    Args:
        names: Experiment identifiers such as ``01_baseline``.

    Returns:
        Instantiated experiment objects.

    Raises:
        ValueError: If any requested experiment is not discoverable.
    """
    available = _get_experiment_prototypes()
    missing = [name for name in names if name not in available]
    if missing:
        known = ", ".join(sorted(available))
        raise ValueError(f"Unknown experiment(s): {missing}. Known experiments: {known}")
    return [_fresh_experiment(available[name]) for name in names]


def _get_experiment_prototypes() -> dict[str, object]:
    """Discover experiments once and reuse prototypes across matrix pairs."""
    global _EXPERIMENT_PROTOTYPES
    if _EXPERIMENT_PROTOTYPES is None:
        from experiments import discover_experiments

        _EXPERIMENT_PROTOTYPES = {
            experiment.meta.name: experiment for experiment in discover_experiments()
        }
    return _EXPERIMENT_PROTOTYPES


def _fresh_experiment(prototype: object) -> object:
    """Create a clean experiment instance from a discovered prototype."""
    try:
        return type(prototype)()
    except TypeError:
        return copy.deepcopy(prototype)


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
    from experiments import ExperimentRunner

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


def run_model_matrix(args: argparse.Namespace) -> list[tuple[ModelPair, list[dict[str, Any]]]]:
    """Run or load every requested model-pair comparison."""
    pair_results: list[tuple[ModelPair, list[dict[str, Any]]]] = []
    pairs = build_model_pairs(args)
    if not pairs:
        raise ValueError("No model pairs selected")

    for pair in pairs:
        pair_output_dir = args.output_dir / pair.slug
        pair_plots_dir = args.plots_dir / f"{pair.slug}-plots"
        pair_args = _args_for_model_pair(args, pair, pair_output_dir, pair_plots_dir)
        print(f"\n### Model pair: {pair.title}")
        print(f"Drafter: {pair.drafter.path}")
        print(f"Target:  {pair.target.path}")

        if args.plot_only:
            results = load_results(pair_output_dir, args.experiments)
        else:
            results = run_comparison(pair_args)

        metrics_csv = pair_output_dir / "metrics.csv"
        write_union_csv(results, metrics_csv)
        write_union_csv(results, pair_output_dir / "dynamic_k_comparison.csv")

        failures = failed_experiments(results)
        if failures and not args.continue_on_error:
            raise RuntimeError(f"Experiment failure(s) for {pair.slug}: " + " | ".join(failures))

        plot_path = plot_model_pair_summary(results, pair_plots_dir, pair)
        print(f"Pair CSV saved to: {metrics_csv}")
        print(f"Pair plot saved to: {plot_path}")
        pair_results.append((pair, results))

    matrix_csv = args.output_dir / "model_matrix_metrics.csv"
    write_matrix_csv(pair_results, matrix_csv)
    print(f"\nMatrix CSV saved to: {matrix_csv}")
    return pair_results


def _args_for_model_pair(
    args: argparse.Namespace,
    pair: ModelPair,
    output_dir: Path,
    plots_dir: Path,
) -> argparse.Namespace:
    pair_args = argparse.Namespace(**vars(args))
    pair_args.output_dir = output_dir
    pair_args.plots_dir = plots_dir
    pair_args.tiny = False
    pair_args.drafter_model = pair.drafter.path
    pair_args.target_model = pair.target.path
    return pair_args


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


def write_matrix_csv(
    pair_results: Sequence[tuple[ModelPair, Sequence[dict[str, Any]]]],
    path: Path,
) -> None:
    """Write one aggregate CSV across all model-pair comparisons."""
    path.parent.mkdir(parents=True, exist_ok=True)
    metric_keys: list[str] = []
    seen: set[str] = set()
    for _, results in pair_results:
        for key in metric_keys_union(results):
            if key not in seen:
                seen.add(key)
                metric_keys.append(key)

    fieldnames = [
        "pair",
        "drafter_size",
        "target_size",
        "drafter_model",
        "target_model",
        "experiment",
        *metric_keys,
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for pair, results in pair_results:
            for result in results:
                metrics = result.get("metrics", {})
                row: dict[str, Any] = {
                    "pair": pair.slug,
                    "drafter_size": pair.drafter.label,
                    "target_size": pair.target.label,
                    "drafter_model": pair.drafter.path,
                    "target_model": pair.target.path,
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


def plot_model_pair_summary(
    results: Sequence[dict[str, Any]],
    plots_dir: Path,
    pair: ModelPair,
) -> Path:
    """Generate one combined comparison PNG for a model pair."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    plt = _load_matplotlib()
    names = _experiment_names(results)
    colors = ["#2563eb", "#16a34a", "#dc2626", "#9333ea", "#f59e0b"]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes_flat = list(axes.reshape(-1))
    fig.suptitle(f"{pair.title}: technique comparison", fontsize=15)

    _bar_metric(axes_flat[0], names, results, "tokens_per_sec", "tokens/sec", colors)
    _bar_metric(axes_flat[1], names, results, "wall_clock_speedup", "speedup vs target", colors)
    _bar_metric(
        axes_flat[2], names, results, "acceptance_rate", "acceptance, %", colors, scale=100.0
    )
    _bar_metric(axes_flat[3], names, results, "wall_time_total_s", "wall time, s", colors)
    _bar_two_metrics(
        axes_flat[4],
        names,
        [_dynamic_mean_k(result) for result in results],
        [_metric_float(result, "avg_accepted_tokens") or 0.0 for result in results],
        "selected k / accepted tokens",
    )
    _plot_regime_distribution_on_axis(axes_flat[5], results)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = plots_dir / "comparison.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _csv_value(value: Any) -> Any:
    if isinstance(value, dict | list):
        return json.dumps(value, sort_keys=True)
    return value


def _slugify(value: str) -> str:
    return value.lower().replace(".", "_").replace("/", "_").replace(" ", "_")


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


def _bar_metric(
    axis,
    names: Sequence[str],
    results: Sequence[dict[str, Any]],
    metric: str,
    ylabel: str,
    colors: Sequence[str],
    *,
    scale: float = 1.0,
) -> None:
    values = [(_metric_float(result, metric) or 0.0) * scale for result in results]
    axis.bar(names, values, color=colors[: len(names)])
    axis.set_title(metric)
    axis.set_ylabel(ylabel)
    axis.tick_params(axis="x", rotation=25)
    for label in axis.get_xticklabels():
        label.set_horizontalalignment("right")


def _bar_two_metrics(
    axis,
    names: Sequence[str],
    left_values: Sequence[float],
    right_values: Sequence[float],
    title: str,
) -> None:
    x_positions = list(range(len(names)))
    width = 0.38
    axis.bar(
        [position - width / 2 for position in x_positions],
        left_values,
        width=width,
        label="selected k / draft length",
        color="#2563eb",
    )
    axis.bar(
        [position + width / 2 for position in x_positions],
        right_values,
        width=width,
        label="avg accepted tokens",
        color="#16a34a",
    )
    axis.set_xticks(x_positions, names, rotation=25, ha="right")
    axis.set_ylabel("tokens")
    axis.set_title(title)
    axis.legend(fontsize=8)


def _plot_regime_distribution_on_axis(axis, results: Sequence[dict[str, Any]]) -> None:
    distributions = _regime_k_distributions(results)
    if not distributions:
        axis.axis("off")
        axis.text(0.5, 0.5, "No regime-k distribution", ha="center", va="center")
        return

    keys = sorted({k for _, distribution in distributions for k in distribution})
    x_positions = list(range(len(keys)))
    width = min(0.8 / max(len(distributions), 1), 0.35)
    colors = ["#9333ea", "#f59e0b", "#0f766e", "#dc2626"]

    for index, (name, distribution) in enumerate(distributions):
        offset = (index - (len(distributions) - 1) / 2) * width
        values = [distribution.get(k, 0.0) for k in keys]
        axis.bar(
            [position + offset for position in x_positions],
            values,
            width=width,
            label=name,
            color=colors[index % len(colors)],
        )

    axis.set_xticks(x_positions, [str(key) for key in keys])
    axis.set_title("Dynamic-k selected-k distribution")
    axis.set_xlabel("selected k")
    axis.set_ylabel("steps")
    axis.legend(fontsize=7)


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
    distributions = _regime_k_distributions(results)
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


def _regime_k_distributions(
    results: Sequence[dict[str, Any]],
) -> list[tuple[str, dict[int, float]]]:
    distributions: list[tuple[str, dict[int, float]]] = []
    for result in results:
        metrics = result.get("metrics", {})
        name = str(result.get("config", {}).get("name", metrics.get("name", "")))
        distribution = metrics.get("regime_k_k_distribution")
        if isinstance(distribution, dict) and distribution:
            normalized = _normalize_distribution(distribution)
            if normalized:
                distributions.append((name, normalized))
    return distributions


def _plot_controller_diagnostics(
    results: Sequence[dict[str, Any]], plots_dir: Path, plt
) -> Path | None:
    diagnostic_keys = (
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
    if args.matrix:
        run_model_matrix(args)
        return

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
