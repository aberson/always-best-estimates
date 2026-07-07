"""One source of truth for data-shape constants (plan.md section 12 Appendix).

Every producer AND consumer imports these values from here — never redefine
them elsewhere. Identity (``is``) regression tests arrive with the first
cross-module consumer (Step 2+).
"""

from typing import Final

HORIZON_BARS: Final[int] = 21  # trading days; the single forecast horizon
TRADING_DAYS: Final[int] = 252  # annualization factor

UNIVERSE: Final[tuple[str, ...]] = ("SPY", "ACWI", "AGG")

# BL equilibrium prior (60/40, equity leg split)
W_MKT: Final[dict[str, float]] = {"SPY": 0.30, "ACWI": 0.30, "AGG": 0.40}

DELTA: Final[float] = 2.5  # He-Litterman risk aversion; shared by BL prior + optimizer
TAU: Final[float] = 0.05  # BL prior scalar (cancels in idzorek omega mode)
W_MAX: Final[float] = 0.60  # per-asset box cap

FRED_DAILY: Final[tuple[str, ...]] = (
    "DGS10",
    "T10Y2Y",
    "VIXCLS",
    "DFF",
    "BAMLH0A0HYM2",
    "DTWEXBGS",
)

# Business days from obs_date to available_date
FRED_RELEASE_LAG: Final[dict[str, int]] = {
    "DGS10": 1,
    "T10Y2Y": 1,
    "VIXCLS": 1,
    "DFF": 1,
    "BAMLH0A0HYM2": 1,
    "DTWEXBGS": 3,
}
