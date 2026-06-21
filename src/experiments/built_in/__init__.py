"""Built-in experiments corresponding to the original ablation suite.

Each module contains one or more ``BaseExperiment`` subclasses that
reproduce the behaviour of the original flag-based ``ABLATION_SUITE``
configs.  Import the classes you need and instantiate them:

>>> from experiments.built_in import (
...     BaselineExperiment,
...     LatticeExperiment,
...     FullSystemExperiment,
... )
>>> experiments = [
...     BaselineExperiment(),
...     LatticeExperiment(),
...     FullSystemExperiment(),
... ]
"""

from .baseline import BaselineExperiment
from .full_system import FullSystemExperiment
from .with_contrastive import ContrastiveExperiment
from .with_lattice import LatticeExperiment
from .with_online_distil import OnlineDistillExperiment
from .with_replay import ReplayExperiment
from .with_routing import RoutingExperiment
from .with_speedup_adapt import SpeedupAdaptiveExperiment
from .with_translator import TranslatorExperiment
from .with_universal import UniversalDrafterExperiment

__all__ = [
    "BaselineExperiment",
    "ContrastiveExperiment",
    "FullSystemExperiment",
    "LatticeExperiment",
    "OnlineDistillExperiment",
    "ReplayExperiment",
    "RoutingExperiment",
    "SpeedupAdaptiveExperiment",
    "TranslatorExperiment",
    "UniversalDrafterExperiment",
]
