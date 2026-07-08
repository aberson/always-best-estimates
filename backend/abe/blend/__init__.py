"""abe.blend — the Black-Litterman blend layer (plan Step 6).

The sigma -> Idzorek confidence map now lives in the transparent ``abe.calc``
module (Track 1 relocation); the blend layer imports it from there.

Modules:
- ``covariance``: Ledoit-Wolf shrinkage as the ONLY annualized-Sigma path.
- ``black_litterman``: ``bl_blend`` — prior pi = delta*Sigma*w_mkt, absolute
  Idzorek views from ``Forecast``s, posterior (mu, Sigma) + diagnostics.
"""
