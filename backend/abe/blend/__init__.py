"""abe.blend — the Black-Litterman blend layer (plan Step 6).

Modules:
- ``covariance``: Ledoit-Wolf shrinkage as the ONLY annualized-Sigma path.
- ``confidence``: the sigma -> Idzorek confidence map (leaf module, no pypfopt).
- ``black_litterman``: ``bl_blend`` — prior pi = delta*Sigma*w_mkt, absolute
  Idzorek views from ``Forecast``s, posterior (mu, Sigma) + diagnostics.
"""
