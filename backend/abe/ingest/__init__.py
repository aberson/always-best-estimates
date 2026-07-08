"""abe.ingest — data-ingest layer (prices via yfinance, macro via fredapi).

Modules:
- ``sources``: ``SourceAdapter`` protocol + ``YFinanceAdapter`` / ``CacheAdapter``.
- ``prices``: incremental price ingest + ``python -m abe.ingest.prices`` CLI.
- ``macro``: FRED macro ingest (arrives in Step 4).
"""
