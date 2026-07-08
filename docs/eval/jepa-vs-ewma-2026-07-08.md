# JEPA vs EWMA walk-forward evaluation (2026-07-08T06:46:43+00:00)

Universe **SPY, ACWI, AGG** | horizon **H = 21** bars | **4** purged walk-forward split(s) | stride **21** (non-overlapping within a fold) | runtime **19.3s**.

## Pre-registered rule (written before results)

Metrics, computed from `(mu, sigma)` produced **through the production `WorldModel.forecast` interface** on identical purged walk-forward windows (each asset's frame is its own full cached history truncated to the eval date — the production `_stage_features` shape): per-asset + pooled forecast **MSE** (mu vs realized H-day log-return), **information coefficient** (Spearman rank corr of mu vs realized, per asset), and **sigma calibration coverage** (fraction with `|realized - mu| <= 1.64 * sigma`; target ~= 0.90).

**Promotion rule (mechanical):** the JEPA is **promoted ONLY IF** (1) its pooled MSE is strictly lower than EWMA's, **AND** (2) its pooled coverage is within +-0.05 of 0.90, **AND** (3) its coverage is no worse than EWMA's distance from 0.90. Otherwise **EWMA remains the default** (EWMA wins ties).

## Decision

### JEPA promoted

Rationale: pooled MSE strictly lower (1.3892e-03 < 1.4292e-03), coverage 0.897 within +-0.05 of 0.90, and no worse than EWMA (|0.897-0.90|=0.003 <= 0.024)

## Pooled metrics

| Model | Version | Points | Pooled MSE | Pooled IC | Coverage |
|---|---|---:|---:|---:|---:|
| EWMA | `ewma` | 525 | 1.4292e-03 | +0.036 | 0.924 |
| JEPA | `per-fold ensembles: jepa:4d203fcb, jepa:248aff0b, jepa:d3a10ee5, jepa:33cb9304` | 525 | 1.3892e-03 | +0.033 | 0.897 |

Pooled MSE is equal-point-weighted across assets (higher-variance SPY/ACWI dominate its magnitude; AGG contributes little). Pooled IC is Spearman over the concatenated cross-asset series — a scale artifact reported for context only; the promotion rule never uses it.

## Per-asset metrics

| Asset | Model | MSE | IC | Coverage |
|---|---|---:|---:|---:|
| SPY | EWMA | 1.9955e-03 | -0.171 | 0.920 |
| SPY | JEPA | 1.9646e-03 | -0.017 | 0.914 |
| ACWI | EWMA | 2.0978e-03 | -0.123 | 0.960 |
| ACWI | JEPA | 2.0238e-03 | +0.028 | 0.914 |
| AGG | EWMA | 1.9428e-04 | -0.002 | 0.891 |
| AGG | JEPA | 1.7913e-04 | +0.026 | 0.863 |

## Folds

| Fold | Train bars | Test bars | Eval points | First eval | Last eval |
|---:|---:|---:|---:|---|---|
| 0 | 899 | 919 | 44 | 2011-11-18 | 2015-06-25 |
| 1 | 1818 | 919 | 44 | 2015-07-20 | 2019-02-20 |
| 2 | 2737 | 919 | 44 | 2019-03-14 | 2022-10-12 |
| 3 | 3656 | 919 | 43 | 2022-11-03 | 2026-05-13 |

## Fingerprints

- **Config** (`f0cce4055242`):

```json
{
  "calibration_z": 1.64,
  "coverage_target": 0.9,
  "coverage_tolerance_pp": 0.05,
  "horizon": 21,
  "jepa_config": {
    "context_len": 16,
    "cov_weight": 0.1,
    "effective_rank_threshold": 1.5,
    "ema_momentum": 0.99,
    "epochs": 200,
    "feature_names": [
      "log_return"
    ],
    "hidden_dim": 64,
    "holdout_fraction": 0.25,
    "horizon": 21,
    "inv_weight": 1.0,
    "latent_dim": 48,
    "lr": 0.001,
    "n_seeds": 3,
    "return_head_hidden": 16,
    "return_weight": 0.05,
    "seed": 0,
    "sigma_floor": 1e-08,
    "std_median_threshold": 0.1,
    "var_weight": 1.0,
    "vicreg_gamma": 1.0
  },
  "n_splits": 4,
  "stride": 21
}
```

- **Data:**

```json
{
  "assets": {
    "ACWI": {
      "max_date": "2026-07-07",
      "rows": 4597
    },
    "AGG": {
      "max_date": "2026-07-07",
      "rows": 5728
    },
    "SPY": {
      "max_date": "2026-07-07",
      "rows": 8415
    }
  },
  "panel_bars": 4596,
  "panel_end": "2026-07-07",
  "panel_start": "2008-03-31"
}
```

## How promotion happens (manual operator action)

This report only **records** the decision. Promotion is a deliberate operator step, never automatic: the scheduler and API default to EWMA and this eval never touches them. To promote the JEPA, set `ABE_MODEL=jepa` + `ABE_JEPA_CHECKPOINT=<path>` at startup (the Step 13 toggle via `resolve_startup_model`). Until an operator flips it, EWMA stays the live forecaster — consistent with plan section 1 (V1 is never gated on the JEPA beating the baseline; on ~4,600 joint bars, parity is the realistic best case).
