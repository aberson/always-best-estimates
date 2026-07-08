"""The sigma -> Idzorek confidence map (plan sections 4 + 8; leaf module).

``c = clamp(|2*Phi(mu/sigma) - 1|, 0.02, 0.95)`` with ``Phi`` the standard
normal CDF. ``|2*Phi(x) - 1|`` is the probability mass of a standard normal
within ``|x|`` of zero — i.e. how sure the model is that the H-day return
shares ``mu``'s sign given predictive uncertainty ``sigma``. A no-information
forecast (``mu ~ 0`` relative to ``sigma``) lands on the 0.02 floor, so
Black-Litterman gracefully degrades to the market prior; a delusionally
certain forecast is capped at 0.95 so the Idzorek omega never collapses to 0
(plan section 9's endpoint risk: c=0 -> omega=1e6, c=1 -> omega=0).

Deliberately a LEAF module: no pypfopt import (mirrors ``constants.py``'s
one-source-of-truth discipline — ``black_litterman.py`` imports the map, tests
import the clamp constants; nobody re-derives them). ``Phi`` comes from
``math.erf`` — no scipy: ``|2*Phi(x) - 1| = erf(|x| / sqrt(2))``.

Inputs are the RAW H-day ``(mu, sigma)`` exactly as a ``Forecast`` emits them
(see ``black_litterman.bl_blend`` for why they are NOT annualized first).
"""

import math
from typing import Final

__all__ = [
    "CONFIDENCE_CEIL",
    "CONFIDENCE_FLOOR",
    "idzorek_confidence",
]

CONFIDENCE_FLOOR: Final[float] = 0.02
"""Lower clamp: keeps the Idzorek omega finite (c=0 would mean omega=1e6 in
pypfopt — an arbitrary sentinel, not math) while still yielding a posterior
that is prior-dominated to ~2 decimal places."""

CONFIDENCE_CEIL: Final[float] = 0.95
"""Upper clamp: c=1 would mean omega=0 (a view treated as certain truth); 0.95
caps any single forecast's pull on the posterior."""


def idzorek_confidence(mu: float, sigma: float) -> float:
    """Map an H-day forecast ``(mu, sigma)`` to an Idzorek view confidence.

    ``c = clamp(|2*Phi(mu/sigma) - 1|, CONFIDENCE_FLOOR, CONFIDENCE_CEIL)``,
    computed as ``erf(|mu/sigma| / sqrt(2))`` (identical by symmetry of the
    normal CDF, so the map is even in ``mu`` by construction).

    Scale note: ``c`` depends only on the RATIO ``mu/sigma``, so it is
    invariant under any rescaling applied to BOTH inputs identically — but the
    H-day -> annual conversion is NOT such a rescaling (``mu`` scales by
    252/H, an independent-draws ``sigma`` by sqrt(252/H)), which is why
    ``bl_blend`` feeds this map the raw H-day pair (decision documented
    there).

    Properties (all pinned by tests):
    - ``mu = 0`` -> exactly ``CONFIDENCE_FLOOR`` (no-information forecast).
    - ``sigma -> 0+`` (or huge ``|mu|``, e.g. the "25 instead of 0.25" misuse)
      -> saturates at ``CONFIDENCE_CEIL``; never overflows or crashes
      (``erf(inf) == 1``).
    - ``sigma`` huge -> floor.
    - Symmetric in the sign of ``mu``.

    Raises ``ValueError`` on non-finite ``mu`` and on ``sigma <= 0`` or
    non-finite ``sigma`` (a ``Forecast`` can never carry those, but this map
    is also callable on raw floats and must fail loud on its own).
    """
    if not math.isfinite(mu):
        raise ValueError(f"mu must be finite, got {mu!r}")
    if not math.isfinite(sigma) or sigma <= 0.0:
        raise ValueError(f"sigma must be finite and > 0 (H-day predictive std), got {sigma!r}")
    raw = math.erf(abs(mu / sigma) / math.sqrt(2.0))
    return min(max(raw, CONFIDENCE_FLOOR), CONFIDENCE_CEIL)
