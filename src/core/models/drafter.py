"""
core/models/drafter.py

Backwards-compatible re-export. Prefer importing from the dedicated
modules instead::

    from core.models.draft_model import DraftModel
    from core.models.target_model import TargetModel
"""

from core.models.draft_model import DraftModel
from core.models.target_model import TargetModel

__all__ = ["DraftModel", "TargetModel"]
