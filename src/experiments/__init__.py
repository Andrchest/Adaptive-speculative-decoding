"""Experiment framework for Adaptive Speculative Decoding.

Public API
----------
BaseExperiment :
    Abstract base class for all experiments.  Subclass this to create
    a new experiment.
ExperimentMeta :
    Metadata attached to every experiment (name, description, tags).
ExperimentResult :
    Immutable result container (meta, config, metrics, error).
BuildContext :
    Shared context passed to ``build_*`` methods.
DecodeContext :
    Mutable context passed to ``on_*`` decode hooks.
ExperimentConfig :
    Dataclass with model paths, dataset, and hyperparameters.
ExperimentRunner :
    Orchestrates a list of experiments and persists results.

Suites
------
ABLATION_SUITE :
    The standard 11-experiment ablation study.
discover_experiments :
    Auto-discover built-in + research experiments.

Quick start
-----------
>>> from experiments import ExperimentRunner, ABLATION_SUITE
>>> runner = ExperimentRunner(experiments=ABLATION_SUITE)
>>> results = runner.run_all()

Custom experiment
-----------------
>>> from experiments import BaseExperiment, ExperimentMeta, ExperimentConfig
>>>
>>> class MyExperiment(BaseExperiment):
...     def __init__(self):
...         super().__init__(ExperimentMeta(name="my_exp"))
...
...     def get_config(self) -> ExperimentConfig:
...         return ExperimentConfig(name="my_exp")
>>>
>>> runner = ExperimentRunner(experiments=[MyExperiment()])
>>> results = runner.run_all()

See ``templates/minimal_template.py`` for a copy-paste starting point.
"""

from .base import BaseExperiment, BuildContext, DecodeContext, ExperimentMeta, ExperimentResult
from .runner import ExperimentConfig, ExperimentRunner
from .suites import ABLATION_SUITE, discover_experiments, discover_research_experiments

__all__ = [
    "ABLATION_SUITE",
    "BaseExperiment",
    "BuildContext",
    "DecodeContext",
    "ExperimentConfig",
    "ExperimentMeta",
    "ExperimentResult",
    "ExperimentRunner",
    "discover_experiments",
    "discover_research_experiments",
]
