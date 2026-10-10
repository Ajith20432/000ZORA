# ZORA Free-AI Rate-Limit Patch

## Defaults
- `MAX_TRADES_PER_DAY=5` (UTC day; entry records are counted once and closing records are not double-counted)
- `MIN_AI_CALL_INTERVAL_SECONDS=900` (global cooldown between AI ensemble invocations)
- `MAX_AI_CALLS_PER_DAY=96` (hard cap on outbound AI-provider HTTP requests, including provider retries, tool rounds, and judge calls)

## Implementation
- `config.py`: adds validated settings and defaults.
- `bot.py`: invokes AI only after a technical-only EMA/RSI/MACD strategy signal is BUY/SELL; enforces the global 15-minute ensemble interval; adds the UTC daily trade cap and Telegram commands `/set_max_trades`, `/ai_usage`, `/pause_ai`, `/resume_ai`.
- `multi_ai.py`: routes outbound provider HTTP calls through the daily-budget gate; preserves provider-local circuit breakers; handles 429 with `Retry-After`/exponential cooldown and falls through to other providers rather than sleeping for minutes in the trading loop.
- `ai_guard.py`: thread-safe daily usage counter persisted beside `DB_PATH`, reset by UTC date, and fails closed if quota persistence fails.
- `telegram_control.py`: publishes the new commands in Telegram's command menu.
- `.env.example`: documents the new defaults.
- `tests/test_ai_usage_budget.py`: verifies quota enforcement and persisted usage restoration.

## Important semantics
- The 900-second interval applies between ensemble invocations, not between individual provider HTTP requests inside one ensemble. This allows parallel multi-provider consensus while the separate hard budget caps total outbound AI requests at 96 per UTC day.
- A 429 response is not immediately retried against the same provider. The provider receives a deferred cooldown based on `Retry-After` and exponential backoff; other healthy providers may still respond. This avoids blocking the trading loop for minutes.
- `/pause_ai` disables AI inference and uses the indicator-based strategy while retaining all hard risk controls. `/resume_ai` resumes rate-limited AI calls.
- `/set_max_trades` changes the limit in memory until restart. The default returns to `MAX_TRADES_PER_DAY` after restart.
- AI usage persists in `<DB_PATH>.ai_usage.json`; protect this file along with the database and environment secrets.

## Validation
- Python syntax compilation passed for `config.py`, `bot.py`, `telegram_control.py`, `multi_ai.py`, and `ai_guard.py`.
- 29 tests passed across configuration, AI guard, Telegram controls, and new AI usage budget tests.
- `tests/test_multi_ai.py` and `tests/test_trader.py` could not be collected in this environment because the `ccxt` package is not installed. Install `requirements.txt` and rerun the full suite before deployment.
