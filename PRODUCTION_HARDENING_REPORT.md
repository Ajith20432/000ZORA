# ZORA Production Hardening Patch

## Included

- Stable SHA-256-derived client order IDs for a persisted logical order key.
- Persistent SQLite `order_journal` and `risk_reservations` tables.
- Atomic `BEGIN IMMEDIATE` risk reservation with idempotent reservation keys.
- No blind retry after network/unknown submission outcomes. Live executor returns explicit `last_outcome` states and attempts client-ID reconciliation when the adapter supports it.
- Startup loads unresolved journal entries, queries the exchange where possible, and blocks new entries for symbols requiring position/protection verification.
- Order and risk state is persisted before live submission; uncertain submissions retain reservations.
- Daily AI budget shared across processes through SQLite, with a rolling-minute request cap and fail-closed behavior.
- Provider-local 429 cooldown with numeric or HTTP-date `Retry-After` handling, bounded fallback, and AI ensemble cycle deadline.
- AI calls remain gated by technical setup and configured minimum interval; daily trade cap and Telegram controls are included.
- Tests for journal persistence, idempotency, risk reservation limits, and no blind network retry.

## Safety semantics

- `UNKNOWN`, `OPEN`, and `PARTIAL` orders are not treated as rejected. Their risk reservations remain active and entries are blocked until reconciliation.
- An order that is found filled during startup is recorded in the journal but does not automatically reconstruct a position's exact stop, take-profit, fees, and protection bracket. ZORA blocks new entries and requires position/protection verification. This avoids inventing exchange state.
- Exchange APIs differ in client-order-ID lookup and historical-order support. Verify the configured CCXT adapter and exchange documentation; if it cannot prove the outcome, the order remains unresolved.
- This patch does not make live trading universally safe. Keep `LIVE_TRADING=false` and validate with paper trading and exchange mocks before considering live use.

## Validation

- `py_compile` passed for the modified runtime modules.
- 141 tests passed across database, executor, AI guard/budget, configuration, Telegram, strategy, risk, indicators, and backtest tests.
- Full exchange-dependent suite could not be collected in this build environment because `ccxt` is not installed and package installation was unavailable. Install dependencies from `requirements.txt` and run the full suite before deployment.
