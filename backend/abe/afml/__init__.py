"""abe.afml — in-project AFML feature transforms (plan.md Step 12).

Kept in-project (plan.md section 2) because the reference charlesrambo repo is
unlicensed and pandas-2-broken. Two leaf modules, no cross-imports; consumers
import directly from the submodules (this package deliberately re-exports
nothing, matching every sibling package).

Modules:

- ``fracdiff`` — fixed-width FFD, min-d ADF search (training folds only),
  frozen ``FracDiffParams``.
- ``purged_cv`` — purged + embargoed chronological walk-forward splits with
  built-in leakage assertions.
"""
