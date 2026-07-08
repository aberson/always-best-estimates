"""abe.eval — the pre-registered JEPA-vs-EWMA walk-forward evaluation gate (plan.md Step 14).

One leaf module, no cross-imports; consumers import directly from the submodule
(this package deliberately re-exports nothing, matching every sibling package):

- ``walk_forward`` — the pre-registered purged walk-forward eval of ``(mu, sigma)``
  produced THROUGH the production ``WorldModel.forecast`` interface, JEPA vs EWMA
  on identical windows, plus the mechanical promotion rule and the committed
  markdown report (``docs/eval/jepa-vs-ewma-<date>.md``).
"""
