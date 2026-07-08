"""abe.optimize — the constrained-optimization layer (plan Step 7).

Modules:
- ``mvu``: ``optimize_weights`` — the hand-rolled cvxpy mean-variance-utility
  QP (``sum_squares(chol.T @ w)``, NEVER ``quad_form``; solver pinned
  CLARABEL; stateful L1 turnover with cold-start drop + INFEASIBLE-retry
  guard) turning the BL posterior into persisted target weights.
"""
