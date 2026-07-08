"""abe.model — the pluggable WorldModel forecaster layer.

Modules:
- ``base``: the frozen ``WorldModel`` Protocol + ``Forecast`` + ``EWMABaseline``.
- ``jepa`` / ``train``: minimal JEPA behind a config toggle (Step 13).

:func:`load_model` is the EWMA<->JEPA toggle factory (plan Step 13). The JEPA
import is LAZY — importing ``abe.model`` or calling ``load_model("ewma")`` never
pulls torch onto the hot EWMA path; only ``load_model("jepa", ...)`` imports the
heavy module.
"""

from pathlib import Path
from typing import Final

from abe.model.base import EWMABaseline, WorldModel

__all__ = ["MODEL_SPECS", "load_model"]

MODEL_SPECS: Final[tuple[str, ...]] = ("ewma", "jepa")
"""The valid ``model_spec`` values :func:`load_model` accepts."""


def load_model(model_spec: str, checkpoint_path: Path | None = None) -> WorldModel:
    """Build the configured ``WorldModel`` (the plan Step 13 EWMA<->JEPA toggle).

    - ``"ewma"`` -> :class:`~abe.model.base.EWMABaseline` (the default;
      ``checkpoint_path`` is ignored).
    - ``"jepa"`` -> :class:`~abe.model.jepa.JEPAModel` loaded from
      ``checkpoint_path`` (required). A missing path, or a missing/malformed
      checkpoint, raises ``ValueError`` — the fail-loud posture (never a silent
      EWMA fallback).

    Any other spec raises ``ValueError``.
    """
    spec = model_spec.strip().lower()
    if spec == "ewma":
        return EWMABaseline()
    if spec == "jepa":
        if checkpoint_path is None:
            raise ValueError(
                "model_spec='jepa' requires a checkpoint_path (ABE_JEPA_CHECKPOINT) — "
                "refusing to fall back to EWMA silently"
            )
        from abe.model.jepa import JEPAModel  # lazy: keep torch off the EWMA path

        return JEPAModel.from_checkpoint(checkpoint_path)
    raise ValueError(f"unknown model_spec {model_spec!r}; expected one of {MODEL_SPECS}")
