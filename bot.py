"""bot.py — ZORA command entry point.

Paper mode is always the default. `live` needs LIVE_TRADING + I_UNDERSTAND_THE_RISK + API keys AND a
typed confirmation. Every entry, paper or live, passes through the same risk engine.
"""
from __future__ import annotations

import argparse
import json
import logging
import signal as _signal
import time
import uuid
from datetime import datetime, timezone

import db
from brain import BRAIN
from config import CFG
from costs import COST_MODEL
from risk import RiskEngine, manage_stop

logger = logging.getLogger("zora.bot")


# --------------------------------------------------------------------------- pure helpers (unit-tested)
def check_exit(position: dict, price: float) -> str | None:
    """'stop' | 'take' | None for the current price (software-monitored levels)."""
    stop, take = position.get("stop"), position.get("take")
    if position["action"] == "BUY":
        if stop is not None and price <= stop:
            return "stop"
        if take is not None and price >= take:
            return "take"
    else:
        if stop is not None and price >= stop:
            return "stop"
        if take is not None and price <= take:
            return "take"
    return None


def settle_trade(p: dict, exit_price: float) -> float:
    """Net PnL of a closed position: price move minus round-trip fees."""
    direction = 1 if p["action"] == "BUY" else -1
    gross = (exit_price - p["entry"]) * p["size"] * direction
    return gross - COST_MODEL.round_trip_fee(p["entry"], exit_price, p["size"])


def sync_exposure(risk: RiskEngine, positions: dict) -> None:
    """Exposure is recomputed from what is actually open, so it can never drift after restarts/failures."""
    risk.open_exposure = sum(CFG.risk_per_trade_pct for p in positions.values() if p)


def restore_state(conn, symbols):
    """Rebuild equity, peak, exposure and open positions from the database after a restart."""
    equity = db.summary(conn)["latest_equity"]
    peak = max(db.peak_equity(conn) or equity, equity)
    risk = RiskEngine(equity)
    risk.peak_equity = peak
    positions: dict[str, dict | None] = {s: None for s in symbols}
    for p in db.load_open_positions(conn):
        if p["symbol"] not in positions:
            logger.warning("open position for %s is not in SYMBOLS; it will still be managed", p["symbol"])
        positions[p["symbol"]] = p
    sync_exposure(risk, positions)
    risk.update_equity(equity)
    return equity, risk, positions


def entry_blockers(*, signal_action: str, paused: bool, live: bool, stale: str | None,
                   quality: tuple[bool, str] | None, degraded: tuple[bool, float] | None,
                   ai_block: tuple[bool, str] | None = None) -> list[str]:
    """Reasons a NEW entry must not happen. Empty list = clear. Guards never block exits."""
    reasons: list[str] = []
    if paused:
        reasons.append("entries paused")
    if stale:
        reasons.append(f"stale data ({stale})")
    if signal_action == "SELL" and live and not CFG.allow_shorts:
        reasons.append("short entries disabled live (ALLOW_SHORTS=false)")
    if quality is not None and not quality[0]:
        reasons.append(f"market quality: {quality[1]}")
    if degraded is not None and degraded[0]:
        reasons.append(f"strategy degraded (rolling PF {degraded[1]:.2f})")
    if ai_block is not None and not ai_block[0] and CFG.ai_entry_block_on_unavailable:
        reasons.append(f"AI safety: {ai_block[1]}")
    return reasons


def doctor() -> int:
    """Offline config / database / brain check. Returns the number of problems found."""
    problems = list(CFG.validate())
    try:
        db.connect().close()
    except Exception as e:  # noqa: BLE001 - report, don't crash the checker
        problems.append(f"database {CFG.db_path} cannot be opened: {e}")
    try:
        BRAIN.snapshot()
    except Exception as e:  # noqa: BLE001
        problems.append(f"brain state unreadable: {e}")
    for p in problems:
        print(" -", p)
    return len(problems)


def ai_available() -> bool:
    return any((CFG.groq_api_key, CFG.openrouter_api_key, CFG.gemini_api_key, CFG.anthropic_api_key, CFG.openai_api_key))


# --------------------------------------------------------------------------- historical replay helpers
def resample_ohlcv(df, tf: str):
    """Aggregate base candles into a higher timeframe (epoch-aligned, UTC)."""
    from guards import timeframe_seconds
    out = (df.set_index("ts").resample(f"{timeframe_seconds(tf)}s", origin="epoch")
           .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
           .dropna().reset_index())
    return out


def make_signal_fn(raw, brain=BRAIN, tail: int = 220):
    """signal_fn(i) for the backtester: technical experts only, no look-ahead (higher timeframes use
    only candles that had already CLOSED at candle i)."""
    import pandas as pd

    from guards import timeframe_seconds
    from strategy import generate_signal
    base_tf = min(CFG.timeframes, key=timeframe_seconds)
    base_secs = timeframe_seconds(base_tf)
    higher = {}
    for tf in CFG.timeframes:
        if timeframe_seconds(tf) > base_secs:
            h = resample_ohlcv(raw, tf)
            closes = pd.DatetimeIndex(h["ts"] + pd.Timedelta(seconds=timeframe_seconds(tf)))
            higher[tf] = (h, closes)

    def signal_fn(i: int):
        ts_end = raw["ts"].iloc[i] + pd.Timedelta(seconds=base_secs)
        mtf = {base_tf: raw.iloc[max(0, i + 1 - tail):i + 1]}
        for tf, (h, closes) in higher.items():
            k = int(closes.searchsorted(ts_end, side="right"))
            mtf[tf] = h.iloc[max(0, k - tail):k]
        return generate_signal(mtf, brain=brain)
    return signal_fn


def _fetch_history(symbol: str, candles: int, tf: str | None = None):
    import data
    from guards import timeframe_seconds
    tf = tf or min(CFG.timeframes, key=timeframe_seconds)
    return data.fetch_ohlcv_history(data.get_exchange(), symbol, tf, candles)


# --------------------------------------------------------------------------- the live/paper trading loop
class Trader:
    def __init__(self, live: bool = False, exchange=None, executor=None):
        import data
        import telegram_control
        from execution import LiveExecutor, PaperExecutor
        self.live = live
        self.tg = telegram_control
        self.conn = db.connect()
        self.equity, self.risk, self.positions = restore_state(self.conn, CFG.symbols)
        self.exchange = exchange if exchange is not None else data.get_exchange(authenticated=live)
        self.executor = executor or (LiveExecutor(self.alert) if live else PaperExecutor())
        self.paused = False
        self.ai_paused = False
        self.max_trades_per_day = int(CFG.max_trades_per_day)
        self.last_ai_call_at = 0.0
        self.poller = self.tg.CommandPoller()
        self.blocked: dict[str, str] = {}
        self.last_signal: dict[str, str] = {}
        self.last_ai: dict[str, tuple[float, float | None, list[str]]] = {}
        self.btc_note = "n/a"
        self.last_day = datetime.now(timezone.utc).date()
        self.last_heartbeat = time.time()
        self.kill_alerted = False
        self.failures = 0
        self.stop_requested = False

    # ----- plumbing
    @property
    def mode(self) -> str:
        return "live" if self.live else "paper"

    def alert(self, msg: str) -> None:
        logger.info(msg)
        self.tg.send(msg)

    def _unrealized(self, prices: dict[str, float]) -> float:
        total = 0.0
        for s, p in self.positions.items():
            if p and s in prices:
                total += (prices[s] - p["entry"]) * p["size"] * (1 if p["action"] == "BUY" else -1)
        return total

    # ----- startup
    def startup(self) -> None:
        self.tg.setup_commands()
        dropped = self.poller.drain()
        if dropped:
            logger.info("discarded %d Telegram command(s) queued while offline", dropped)
        if self.live and CFG.reconcile_on_start:
            # Any order left unresolved by a prior process must be reconciled before entries.
            try:
                for intent in db.load_unresolved_orders(self.conn):
                    symbol = intent["symbol"]
                    try:
                        reconciliation = self.executor.reconcile_order(self.exchange, intent)
                    except Exception as exc:
                        logger.exception("order reconciliation failed for %s", intent["client_order_id"])
                        reconciliation = {"state": "UNKNOWN", "reason": str(exc), "order": None}
                    state, order = reconciliation.get("state"), reconciliation.get("order") or {}
                    if state in ("REJECTED", "CANCELED"):
                        db.update_order_intent(self.conn, intent["order_key"], status=state,
                                               exchange_order_id=order.get("id"),
                                               filled_amount=order.get("filled"),
                                               average_fill_price=order.get("average"),
                                               last_error=reconciliation.get("reason"))
                        try:
                            db.set_reservation_status(self.conn, intent["order_key"], "released")
                        except KeyError:
                            pass
                        continue
                    if state in ("FILLED", "PARTIAL", "OPEN"):
                        db.update_order_intent(self.conn, intent["order_key"], status=state,
                                               exchange_order_id=order.get("id"),
                                               filled_amount=order.get("filled"),
                                               average_fill_price=order.get("average"),
                                               fee_cost=(order.get("fee") or {}).get("cost"),
                                               fee_currency=(order.get("fee") or {}).get("currency"),
                                               last_error="located on exchange; position/protection reconciliation required")
                    self.blocked[symbol] = f"order {intent['client_order_id']} reconciliation={state}; operator/recovery verification required"
                    self.alert(f"[live] CRITICAL {symbol}: persisted order {intent['client_order_id']} reconciliation={state}; new entries blocked until position and protection are verified")
            except Exception:
                logger.exception("could not load order journal; fail-closed by blocking configured symbols")
                for symbol in self.positions:
                    self.blocked[symbol] = "order journal unavailable; reconciliation required"
            for symbol in self.positions:
                r = self.executor.reconcile(self.exchange, symbol, self.positions[symbol])
                if not r.get("safe"):
                    self.blocked[symbol] = r.get("reason", "reconcile failed")
                    self.alert(f"[live] {symbol} entries BLOCKED: {self.blocked[symbol]}")
                    continue
                pos = r.get("position") or {}
                held = float(pos.get("contracts") or 0)
                tracked = self.positions[symbol]
                if tracked is None and held > 0 and not pos.get("spot_balance"):
                    self.blocked[symbol] = "exchange shows a position the bot is not tracking"
                    self.alert(f"[live] {symbol} entries BLOCKED: {self.blocked[symbol]}")
                if tracked is not None and held <= 0:
                    self.alert(f"[live] {symbol}: tracked position not found on exchange; verify manually")
        self.alert(f"ZORA {self.mode} started | {CFG.exchange_id} | {','.join(CFG.symbols)} | equity {self.equity:.2f}")

    # ----- one cycle
    def cycle(self) -> None:
        import data
        self._daily_rollover()
        btc = None
        if CFG.btc_context_symbol and CFG.btc_context_symbol not in self.positions:
            try:
                from guards import timeframe_seconds
                tf = min(CFG.timeframes, key=timeframe_seconds)
                btc = data.fetch_ohlcv_df(self.exchange, CFG.btc_context_symbol, tf)
                chg = (btc["close"].iloc[-1] / btc["close"].iloc[-24] - 1) * 100 if len(btc) > 24 else 0.0
                self.btc_note = f"{CFG.btc_context_symbol} {btc['close'].iloc[-1]:.2f} ({chg:+.2f}% / 24 candles)"
            except Exception as e:  # noqa: BLE001 - context only
                logger.warning("BTC context unavailable: %s", e)
        prices: dict[str, float] = {}
        for symbol in list(self.positions):
            try:
                price = self._process_symbol(symbol, btc)
                if price is not None:
                    prices[symbol] = price
                self.failures = 0
            except Exception:  # noqa: BLE001 - one bad symbol must not stop the loop
                self.failures += 1
                logger.exception("cycle failed for %s", symbol)
                if self.failures % 5 == 0:
                    self.alert(f"[{self.mode}] {self.failures} consecutive cycle failures; check logs/network")
        self._finish_cycle(prices)

    def _finish_cycle(self, prices: dict[str, float]) -> None:
        self.risk.update_equity(self.equity + self._unrealized(prices))
        db.log_equity(self.conn, self.equity)       # realized equity; open PnL is only tracked in memory
        if self.risk.kill_switch_tripped and not self.kill_alerted:
            self.kill_alerted = True
            self.alert(f"[{self.mode}] KILL SWITCH tripped (equity {self.risk.equity:.2f}, peak {self.risk.peak_equity:.2f}). "
                       "No new entries until restart; open positions stay managed by their stops.")
        self._publish()
        hb = CFG.telegram_heartbeat_minutes
        if hb and time.time() - self.last_heartbeat >= hb * 60:
            self.last_heartbeat = time.time()
            self.tg.send(self._cmd_status())

    def _daily_rollover(self) -> None:
        today = datetime.now(timezone.utc).date()
        if today != self.last_day:
            self.last_day = today
            self.risk.reset_daily()

    def _process_symbol(self, symbol: str, btc) -> float | None:
        import data
        import guards
        from indicators import enrich
        mtf = data.fetch_multi_timeframe(self.exchange, symbol, CFG.timeframes)
        base_tf = min(CFG.timeframes, key=guards.timeframe_seconds)
        base = enrich(mtf[base_tf])
        stale, age = guards.is_stale(mtf[base_tf], base_tf, CFG.stale_data_max_candles)
        price, atr = float(base["close"].iloc[-1]), float(base["atr14"].iloc[-1])
        stale_txt = f"{age / 60:.0f} min old" if stale else None
        if stale:
            logger.warning("%s data is stale (%s); skipping this cycle", symbol, stale_txt)
            return None
        pos = self.positions.get(symbol)
        if pos:
            self._manage_position(symbol, pos, price, atr)
        if self.positions.get(symbol) is None:
            self._consider_entry(symbol, mtf, base, price, atr, stale_txt, btc)
        return price

    # ----- exits
    @staticmethod
    def _native(pos: dict) -> bool:
        prot = pos.get("protection") or {}
        return any(isinstance(v, dict) and v.get("id") for v in prot.values())

    def _manage_position(self, symbol: str, pos: dict, price: float, atr: float) -> None:
        if self.live:
            hit = self.executor.protection_triggered(self.exchange, symbol, pos.get("protection"))
            if hit:
                kind, fill = hit
                prot = pos.get("protection") or {}
                sibling = "take" if kind == "stop" else "stop"      # the other leg must not survive and re-open a position
                self.executor.cancel_protection(self.exchange, symbol, {sibling: prot.get(sibling)})
                level = pos.get("stop") if kind == "stop" else pos.get("take")
                self._finalize(symbol, pos, float(fill) if fill else float(level), f"{kind}-native")
                return
        reason = check_exit(pos, price)
        if reason:
            self._close(symbol, pos, price, reason)
            return
        if not self._native(pos):
            new_stop = manage_stop(pos, price, atr)
            if new_stop is not None:
                pos["stop"] = new_stop
                db.save_open_position(self.conn, pos)
                logger.info("%s stop moved to %.4f", symbol, new_stop)

    def _close(self, symbol: str, pos: dict, price: float, reason: str) -> bool:
        side = "sell" if pos["action"] == "BUY" else "buy"
        if self.live:
            order_key = f"exit:{self.mode}:{symbol}:{int(pos.get('opened_ts', 0))}:{reason}"
            client_id = self.executor.client_order_id(order_key)
            try:
                db.create_order_intent(self.conn, order_key=order_key, client_order_id=client_id, symbol=symbol,
                                       side=side, order_type="market", amount=pos["size"], price=price)
                db.update_order_intent(self.conn, order_key, status="SUBMITTING")
                fill = self.executor.close(self.exchange, symbol, pos["action"], pos["size"], price, order_key=order_key)
                if fill is None:
                    uncertain = self.executor.last_outcome in {"UNKNOWN", "OPEN", "PARTIAL", "SUBMITTING"}
                    state = self.executor.last_outcome if uncertain else "REJECTED"
                    db.update_order_intent(self.conn, order_key, status=state, last_error=self.executor.last_outcome)
                    if uncertain:
                        self.blocked[symbol] = f"exit order {client_id} outcome unknown; reconcile before retry"
                        self.alert(f"[live] CRITICAL {symbol} exit outcome unknown; no automatic duplicate retry")
                    else:
                        self.alert(f"[live] {symbol} {reason} exit rejected; inspect exchange before retry")
                    return False
                order = self.executor.last_order or {}
                db.update_order_intent(self.conn, order_key, status="FILLED" if self.executor.last_outcome == "FILLED" else "OPEN",
                                       exchange_order_id=order.get("id"), filled_amount=order.get("filled"),
                                       average_fill_price=order.get("average"), last_error=None)
                self.executor.cancel_protection(self.exchange, symbol, pos.get("protection"))
            except Exception:
                logger.exception("live exit submission failed for %s", symbol)
                self.blocked[symbol] = "exit submission uncertain; reconcile before retry"
                return False
        else:
            if reason == "stop":      # a gap through the stop fills at the (worse) market price
                ref = min(price, pos["stop"]) if pos["action"] == "BUY" else max(price, pos["stop"])
            else:
                ref = pos["take"] if reason == "take" else price
            fill = COST_MODEL.fill_price(ref, side)
        self._finalize(symbol, pos, float(fill), reason)
        return True

    def _finalize(self, symbol: str, pos: dict, exit_price: float, reason: str) -> None:
        pnl = settle_trade(pos, exit_price)
        db.log_trade(self.conn, symbol, pos["action"], exit_price, pos["size"], pos.get("stop"), pos.get("take"),
                     self.mode, result=reason, pnl=pnl)
        db.delete_open_position(self.conn, symbol)
        reservation_id = pos.get("risk_reservation_id")
        if reservation_id:
            try:
                db.set_reservation_status(self.conn, reservation_id, "released")
            except KeyError:
                logger.warning("risk reservation %s missing while finalizing %s", reservation_id, symbol)
        self.positions[symbol] = None
        self.equity += pnl
        self.risk.update_equity(self.equity)
        sync_exposure(self.risk, self.positions)
        BRAIN.update_from_trade(pos.get("expert_scores") or {}, pos.get("factor_scores_by_tf") or {}, pos["action"], pnl)
        db.log_equity(self.conn, self.equity)
        self.alert(f"[{self.mode}] CLOSED {pos['action']} {symbol} @ {exit_price:.4f} ({reason}) PnL {pnl:+.2f} | equity {self.equity:.2f}")

    # ----- entries
    def _experts(self, symbol: str, mtf: dict, base_raw):
        import fear_greed
        import ml_model
        import news_feed
        import tv_signal
        extra: dict = {}
        notes: list[str] = []
        headlines: list[str] = []
        try:
            score, headlines = news_feed.news_sentiment(symbol)
            if headlines:
                extra["news_score"] = score
        except Exception as e:  # noqa: BLE001
            logger.warning("news expert failed: %s", e)
        if CFG.fear_greed_enabled:
            try:
                score, note = fear_greed.fear_greed_score()
                if score is not None:
                    extra["fng_score"] = score
                    notes.append(note)
            except Exception as e:  # noqa: BLE001
                logger.warning("fear&greed expert failed: %s", e)
        try:
            res = ml_model.ml_expert(base_raw)
            if res.prob_up is not None:     # only vote when the model has proven an edge out-of-sample
                extra["ml_score"] = res.score
        except Exception as e:  # noqa: BLE001
            logger.warning("ml expert failed: %s", e)
        score, note = tv_signal.current_score()
        if score is not None:
            extra["tv_score"] = score
            notes.append(f"tv: {note}")
        return extra, headlines, notes

    def _ai(self, symbol: str, pre_signal, headlines: list[str]):
        """Call the ensemble only for a technical BUY/SELL setup, with a global cooldown."""
        if self.ai_paused:
            return None, ["AI paused by administrator; indicator-only mode"]
        if getattr(pre_signal, "action", "HOLD") not in {"BUY", "SELL"}:
            return None, ["AI skipped: EMA/RSI/MACD technical setup not detected"]
        if not ai_available():
            return None, ["AI skipped: no provider configured; technical-only mode"]

        now = time.time()
        elapsed = now - self.last_ai_call_at
        if self.last_ai_call_at and elapsed < CFG.min_ai_call_interval_seconds:
            wait = int(CFG.min_ai_call_interval_seconds - elapsed)
            return None, [f"AI skipped: global {CFG.min_ai_call_interval_seconds}s interval; {wait}s remaining"]

        import multi_ai
        # Record the start before calling, so failures/timeouts also respect the
        # 15-minute interval and cannot cause a request storm on every loop.
        self.last_ai_call_at = now
        try:
            score, notes = multi_ai.get_ai_consensus(symbol, pre_signal, headlines, self.exchange, self.btc_note)
        except Exception as e:  # AI is advisory; failures never crash the trading cycle.
            logger.warning("AI panel failed: %s", e)
            score, notes = None, [f"AI error; technical fallback policy applies: {e}"]
        self.last_ai[symbol] = (time.time(), score, notes)
        return score, notes

    def _trades_opened_today(self) -> int:
        """Count entry records in UTC today; closing records are not double-counted."""
        start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
        row = self.conn.execute(
            "SELECT COUNT(*) FROM trades WHERE ts >= ? AND mode = ? AND result = 'open'",
            (start, self.mode),
        ).fetchone()
        return int(row[0] if row else 0)

    def _consider_entry(self, symbol, mtf, base, price, atr, stale_txt, btc) -> None:
        import guards
        from strategy import generate_signal
        extra, headlines, notes = self._experts(symbol, mtf, mtf[min(mtf, key=guards.timeframe_seconds)])
        # Gate AI strictly on the EMA/RSI/MACD strategy signal, not on news,
        # ML, fear/greed, or other optional expert votes.
        technical_setup = generate_signal(mtf)
        ai_score, ai_notes = self._ai(symbol, technical_setup, headlines)
        signal = generate_signal(mtf, ai_score=ai_score, **extra)
        self.last_signal[symbol] = f"{signal.action} {signal.confidence:.0f}% [{', '.join(signal.reasons)}]"
        # Preserve provenance for the eventual fill alert. A non-None score is
        # the aggregate returned by multi_ai, not a claim that one provider alone
        # caused the trade. If AI is unavailable, say so explicitly.
        provider_names = []
        for item in ai_notes:
            if ":" in item:
                candidate = item.split(":", 1)[0].strip()
                if candidate and candidate.lower() not in {"ai", "error", "judge synthesis"}:
                    provider_names.append(candidate)
        decision_model = signal.decision_model
        if ai_score is not None:
            ai_rationale = next((item for item in ai_notes if item.strip()), "Providers returned a consensus score.")
        else:
            ai_rationale = next((item for item in signal.reasons if item.strip()), "Technical factors determined the signal.")
        if signal.action == "HOLD":
            return
        note = " | ".join(notes + ai_notes[:2])[:500] or None

        quality = None
        if self.live:
            try:
                quality = guards.market_quality(self.exchange.fetch_ticker(symbol), CFG.max_spread_bps, CFG.min_quote_volume)
            except Exception as e:  # noqa: BLE001 - fail safe: unknown quality means no entry
                quality = (False, f"ticker unavailable: {e}")
        degraded = guards.strategy_degraded(db.recent_closed_pnls(self.conn, CFG.degradation_window),
                                            CFG.degradation_window, CFG.degradation_min_profit_factor)
        import ai_guard
        if self.ai_paused:
            ai_block = (True, "AI paused; indicator-only mode")
        elif ai_score is not None:
            # A successful consensus remains valid for this decision even if
            # its final request consumed the last daily quota slot.
            ai_block = (True, "AI consensus returned for this setup")
        else:
            ai_block = ai_guard.before_cycle()
        blockers = entry_blockers(signal_action=signal.action, paused=self.paused, live=self.live,
                                  stale=stale_txt, quality=quality, degraded=degraded, ai_block=ai_block)
        trades_today = self._trades_opened_today()
        if trades_today >= self.max_trades_per_day:
            blockers.append(f"daily trade limit reached ({trades_today}/{self.max_trades_per_day}, UTC)")
        if symbol in self.blocked:
            blockers.append(f"reconcile: {self.blocked[symbol]}")
        if blockers:
            db.log_decision(self.conn, symbol, signal, False, "; ".join(blockers), note)
            return

        size_mult = 1.0
        if btc is not None and symbol != CFG.btc_context_symbol:
            corr = guards.rolling_correlation(base["close"], btc["close"], CFG.correlation_window)
            if guards.correlation_regime(corr) == "broken":
                size_mult = CFG.correlation_size_mult
        decision = self.risk.evaluate(signal.action, price, atr, size_mult)
        db.log_decision(self.conn, symbol, signal, decision.approved, decision.reason, note)
        if decision.approved:
            decision_context = {
                "decision_model": decision_model,
                "providers": provider_names,
                "ai_consensus": ai_score,
                "rationale": ai_rationale,
            }
            self._open(symbol, signal, decision, price, decision_context)

    def _open(self, symbol: str, signal, decision, price: float, decision_context: dict | None = None) -> None:
        action, size = signal.action, decision.position_size
        if self.live:
            if CFG.native_protection_required and not self.executor._native_supported(self.exchange, symbol):
                self.alert(f"[live] {symbol} entry refused: native stop/take-profit not confirmed for this market")
                return
            r = self.executor.reconcile(self.exchange, symbol)
            if not r.get("safe") or r.get("open_orders"):
                self.alert(f"[live] {symbol} entry refused: exchange state not clean ({r.get('reason', 'open orders exist')})")
                return
            order_key = f"entry:{self.mode}:{symbol}:{uuid.uuid4().hex}"
            client_id = self.executor.client_order_id(order_key)
            reserved_notional = max(0.0, float(price) * float(size))
            max_reserved = max(0.01, float(self.equity) * float(CFG.max_exposure_pct) / 100.0)
            if not db.reserve_risk(self.conn, order_key, symbol, self.mode, reserved_notional, max_reserved):
                self.alert(f"[live] {symbol} entry refused: atomic risk reservation would exceed exposure limit")
                return
            try:
                db.create_order_intent(self.conn, order_key=order_key, client_order_id=client_id, symbol=symbol,
                                       side="buy" if action == "BUY" else "sell", order_type="market",
                                       amount=size, price=price)
                db.update_order_intent(self.conn, order_key, status="SUBMITTING")
                fill = self.executor.open(self.exchange, symbol, action, size, price, order_key=order_key)
            except Exception:
                logger.exception("entry order failed after intent persistence for %s", symbol)
                db.update_order_intent(self.conn, order_key, status="UNKNOWN", last_error="exception during submission")
                db.set_reservation_status(self.conn, order_key, "uncertain")
                self.blocked[symbol] = f"entry {client_id} outcome uncertain; reconcile before retry"
                self.alert(f"[live] CRITICAL {symbol}: entry outcome uncertain; risk remains reserved")
                return
            if fill is None:
                uncertain = self.executor.last_outcome in {"UNKNOWN", "OPEN", "PARTIAL", "SUBMITTING"}
                order_state = self.executor.last_outcome if uncertain else "REJECTED"
                db.update_order_intent(self.conn, order_key, status=order_state, last_error=self.executor.last_outcome)
                db.set_reservation_status(self.conn, order_key, "uncertain" if uncertain else "released")
                if uncertain:
                    self.blocked[symbol] = f"entry {client_id} outcome uncertain; reconcile before retry"
                    self.alert(f"[live] CRITICAL {symbol}: entry outcome uncertain; no duplicate retry will be sent")
                return
            order = self.executor.last_order or {}
            db.update_order_intent(self.conn, order_key, status="FILLED" if self.executor.last_outcome == "FILLED" else "OPEN",
                                   exchange_order_id=order.get("id"), filled_amount=order.get("filled"),
                                   average_fill_price=order.get("average"), last_error=None)
            db.set_reservation_status(self.conn, order_key, "committed")
            size = self.executor.last_amount or size        # exchange-precision amount of the most recent order actually sent
        else:
            fill = COST_MODEL.fill_price(price, "buy" if action == "BUY" else "sell")
        pos = {"symbol": symbol, "action": action, "entry": float(fill), "size": size, "stop": decision.stop_loss,
               "take": decision.take_profit, "initial_stop": decision.stop_loss, "mode": self.mode, "protection": {},
               "expert_scores": signal.expert_scores, "factor_scores_by_tf": signal.factor_scores_by_tf,
               "opened_ts": time.time(), "risk_reservation_id": order_key if self.live else None}
        if self.live:
            prot = self.executor.place_protection(self.exchange, symbol, action, size, decision.stop_loss, decision.take_profit)
            if prot is None:                       # never stay in a live trade without protection
                self.alert(f"[live] {symbol} protection failed; closing the new position immediately")
                if self.executor.close(self.exchange, symbol, action, size, price) is not None:
                    self._finalize(symbol, pos, float(fill), "protection_failed")
                    return
                self.alert(f"[live] CRITICAL {symbol}: could not close an unprotected position. Close it manually NOW.")
                prot = {"software_only": True}
            pos["protection"] = prot
        self.positions[symbol] = pos
        db.save_open_position(self.conn, pos)
        db.log_trade(self.conn, symbol, action, float(fill), size, decision.stop_loss, decision.take_profit, self.mode)
        self.risk.add_exposure(decision.exposure_pct)
        sync_exposure(self.risk, self.positions)
        context = decision_context or {}
        try:
            self.tg.trade_alert(
                symbol=symbol,
                action=action,
                mode=self.mode,
                quantity=float(size),
                fill_price=float(fill),
                stop_loss=decision.stop_loss,
                take_profit=decision.take_profit,
                decision_model=context.get("decision_model", "ZORA Strategy Engine (provenance unavailable)"),
                strategy_confidence=float(signal.confidence),
                ai_consensus=context.get("ai_consensus"),
                providers=context.get("providers", []),
                rationale=str(context.get("rationale", "No rationale supplied.")),
            )
        except Exception:
            # Notification failure must never undo or interrupt a confirmed trade.
            logger.exception("Could not send formatted trade alert for %s", symbol)
            self.alert(f"[{self.mode}] OPENED {action} {symbol} {size:.6f} @ {fill:.4f} SL {decision.stop_loss:.4f} TP {decision.take_profit:.4f}")

    # ----- dashboard / telegram
    def _publish(self) -> None:
        try:
            import dashboard
            dashboard.publish(mode=self.mode, paused=self.paused, kill_switch=self.risk.kill_switch_tripped,
                              equity=round(self.equity, 2), exposure_pct=round(self.risk.open_exposure, 2),
                              signals=dict(self.last_signal), btc=self.btc_note, updated=time.time())
        except Exception:  # noqa: BLE001 - the dashboard is optional
            pass

    def handle_command(self, text: str) -> str:
        cmd = text.split()[0].lower().split("@")[0]
        if cmd in ("/start", "/help"):
            return ("Commands: /status /signal /btc /ai /ai_usage /pause_ai /resume_ai "
                    "/set_max_trades <number> /brain /pnl /positions /metrics /guards /pause /resume /kill")
        if cmd == "/status":
            return self._cmd_status()
        if cmd == "/signal":
            return "\n".join(f"{s}: {v}" for s, v in self.last_signal.items()) or "no signal computed yet"
        if cmd == "/btc":
            return self.btc_note
        if cmd == "/ai":
            if not ai_available():
                return "no AI provider keys configured"
            import multi_ai
            ok, msg = multi_ai.groq_healthcheck()
            return f"groq: {msg}" + ("" if ok or CFG.groq_api_key else "")
        if cmd == "/brain":
            return json.dumps(BRAIN.snapshot(), indent=1)
        if cmd == "/pnl":
            s = db.summary(self.conn)
            return f"closed trades {s['closed_trades']} | realized PnL {s['realized_pnl']:+.2f} | equity {self.equity:.2f}"
        if cmd == "/positions":
            lines = [f"{p['action']} {s} {p['size']:.6f} @ {p['entry']:.4f} SL {p['stop']:.4f} TP {p['take']:.4f}"
                     for s, p in self.positions.items() if p]
            return "\n".join(lines) or "no open positions"
        if cmd == "/metrics":
            from metrics import compute_metrics, format_metrics
            return format_metrics(compute_metrics(db.recent_closed_pnls(self.conn, 500), None, CFG.starting_capital))
        if cmd == "/guards":
            return (f"paused={self.paused} kill={self.risk.kill_switch_tripped} exposure={self.risk.open_exposure:.2f}/"
                    f"{CFG.max_exposure_pct} blocked={self.blocked or 'none'} | {self.btc_note}")
        if cmd == "/pause":
            self.paused = True
            return "new entries paused (open positions stay managed)"
        if cmd == "/resume":
            self.paused = False
            return "new entries resumed" + (" — but the kill switch is still tripped" if self.risk.kill_switch_tripped else "")
        if cmd == "/kill":
            self.risk.kill_switch_tripped = True
            self.paused = True
            return "KILL SWITCH tripped: no new entries until the bot is restarted. Open positions stay managed by their stops."
        return "unknown command; try /help"

    def _cmd_status(self) -> str:
        open_n = sum(1 for p in self.positions.values() if p)
        return (f"ZORA {self.mode} | equity {self.equity:.2f} (peak {self.risk.peak_equity:.2f}) | open {open_n} | "
                f"trades today {self._trades_opened_today()}/{self.max_trades_per_day} UTC | "
                f"AI paused={self.ai_paused} | exposure {self.risk.open_exposure:.2f}% | "
                f"paused={self.paused} kill={self.risk.kill_switch_tripped}")

    def _handle_commands(self) -> None:
        for text in self.poller.poll():
            try:
                self.tg.send(self.handle_command(text))
            except Exception as e:  # noqa: BLE001
                logger.warning("command %r failed: %s", text, e)

    # ----- main loop
    def _sleep(self, seconds: float) -> None:
        end = time.monotonic() + seconds
        while not self.stop_requested and time.monotonic() < end:
            time.sleep(min(2.0, max(0.0, end - time.monotonic())))
            self._handle_commands()

    def run(self, once: bool = False) -> None:
        self.startup()
        try:
            while not self.stop_requested:
                self._handle_commands()
                self.cycle()
                if once:
                    break
                self._sleep(CFG.poll_seconds)
        except KeyboardInterrupt:
            pass
        finally:
            self.alert(f"ZORA {self.mode} stopped. Open positions: {[s for s, p in self.positions.items() if p] or 'none'}")


# --------------------------------------------------------------------------- CLI commands
def _symbol_arg(args) -> str:
    return args.symbol or CFG.symbol


def cmd_backtest(args, learn: bool = False) -> int:
    from backtester import simulate
    from metrics import compute_metrics, format_metrics
    symbol = _symbol_arg(args)
    from indicators import enrich
    raw = _fetch_history(symbol, args.candles)
    df = enrich(raw)
    sig = make_signal_fn(raw, brain=BRAIN)    # backtest reads the learned weights but never updates them
    hook = (lambda p, pnl: BRAIN.update_from_trade(p["expert_scores"], p["factor_scores_by_tf"], p["action"], pnl)) if learn else None
    res = simulate(df, sig, on_close=hook)
    m = compute_metrics(res.pnls, res.equity_curve, CFG.starting_capital)
    print(f"{'learn' if learn else 'backtest'} {symbol} {len(df)} candles | final equity {res.final_equity:.2f} "
          f"| kill switch {'TRIPPED' if res.kill_switch_tripped else 'ok'}")
    print(format_metrics(m))
    if learn:
        print("brain updated:", json.dumps(BRAIN.snapshot()))
    return 0


def cmd_optimize(args) -> int:
    from indicators import enrich
    from optimize import optimize, save_report
    symbol = _symbol_arg(args)
    raw = _fetch_history(symbol, args.candles)
    df = enrich(raw)
    base_fn, cache = make_signal_fn(raw), {}

    def sig(i):
        if i not in cache:
            cache[i] = base_fn(i)
        return cache[i]
    report = optimize(df, sig)
    save_report(report)
    print(json.dumps(report, indent=2))
    print("report saved to optimize_report.json (nothing was changed in .env)")
    return 0


def cmd_dca(args) -> int:
    from dca import simulate_dca
    symbol = _symbol_arg(args)
    df = _fetch_history(symbol, args.candles)
    r = simulate_dca(df)
    bh = (df["close"].iloc[-1] / df["close"].iloc[0] - 1) * 100
    print(f"DCA {symbol}: cycles={len(r.cycles)} final equity {r.final_equity:.2f} "
          f"({(r.final_equity / CFG.starting_capital - 1) * 100:+.2f}%) max capital used {r.max_capital_used_pct:.1f}% | buy&hold {bh:+.2f}%")
    return 0


def cmd_grid(args) -> int:
    from grid import simulate_grid
    symbol = _symbol_arg(args)
    df = _fetch_history(symbol, args.candles)
    r = simulate_grid(df)
    print(f"GRID {symbol}: round trips {r.round_trips} realized {r.realized_pnl:+.2f} unrealized {r.unrealized_pnl:+.2f} "
          f"fees {r.fees:.2f} stopped={r.stopped} final equity {r.final_equity:.2f} | buy&hold {r.buy_and_hold_pct:+.2f}%")
    return 0


def cmd_metrics(_args) -> int:
    from metrics import compute_metrics, format_metrics
    conn = db.connect()
    pnls = db.recent_closed_pnls(conn, 5000)
    curve = [e for _, e in db.equity_series(conn, 5000)]
    print(format_metrics(compute_metrics(pnls, curve or None, CFG.starting_capital)))
    return 0


def cmd_status(_args) -> int:
    conn = db.connect()
    s = db.summary(conn)
    print(f"equity {s['latest_equity']:.2f} | closed trades {s['closed_trades']} | realized PnL {s['realized_pnl']:+.2f}")
    for p in db.load_open_positions(conn):
        print(f"  open {p['action']} {p['symbol']} {p['size']:.6f} @ {p['entry']:.4f} SL {p['stop']} TP {p['take']} ({p['mode']})")
    for t in db.recent_trades(conn, 5):
        print(f"  {time.strftime('%Y-%m-%d %H:%M', time.localtime(t['ts']))} {t['symbol']} {t['action']} {t['result']} pnl={t['pnl']}")
    return 0


def cmd_run(args, live: bool) -> int:
    problems = CFG.validate()
    if live:
        if not CFG.live_allowed():
            print("LIVE is locked. Needs LIVE_TRADING=true, I_UNDERSTAND_THE_RISK=true and EXCHANGE_API_KEY/SECRET.")
            return 2
        if problems:
            print("Refusing to go live with configuration problems:")
            for p in problems:
                print(" -", p)
            return 2
        print(f"LIVE on {CFG.exchange_id}: symbols={','.join(CFG.symbols)} risk/trade={CFG.risk_per_trade_pct}% "
              f"max exposure={CFG.max_exposure_pct}% daily loss={CFG.max_daily_loss_pct}% drawdown={CFG.max_drawdown_pct}%")
        if input("Type I ACCEPT THE RISK to continue: ").strip() != "I ACCEPT THE RISK":
            print("Aborted.")
            return 2
    else:
        for p in problems:
            print("warning:", p)
    try:
        from dashboard import start_background
        start_background()
    except ImportError:
        logger.warning("dashboard unavailable (fastapi not installed); continuing without it")
    trader = Trader(live=live)
    _signal.signal(_signal.SIGTERM, lambda *_: setattr(trader, "stop_requested", True))
    trader.run(once=args.once)
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="bot.py", description="ZORA trading bot (paper by default)")
    sub = ap.add_subparsers(dest="command", required=True)
    for name in ("backtest", "learn", "optimize", "dca", "grid", "portfolio-backtest", "paper-report", "replay-report"):
        p = sub.add_parser(name)
        p.add_argument("symbol", nargs="?", default=None)
        p.add_argument("candles", nargs="?", type=int, default=1500)
    for name in ("run", "live"):
        p = sub.add_parser(name)
        p.add_argument("--once", action="store_true", help="run a single cycle and exit")
    for name in ("doctor", "status", "brain", "metrics", "dashboard", "ai-health", "ai-test"):
        sub.add_parser(name)
    for name in ("validate", "exchange-smoke", "exchange-sandbox"):
        sub.add_parser(name, add_help=False)      # extra flags are forwarded to the module
    return ap


def main(argv: list[str] | None = None) -> int:
    from logging_setup import configure_logging
    configure_logging()
    args, rest = build_parser().parse_known_args(argv)
    c = args.command
    if c not in ("validate", "exchange-smoke", "exchange-sandbox", "portfolio-backtest", "paper-report", "replay-report") and rest:
        build_parser().error(f"unrecognized arguments: {' '.join(rest)}")
    if c == "doctor":
        n = doctor()
        print("ZORA doctor:", "OK" if n == 0 else f"{n} issue(s) found")
        return 0 if n == 0 else 1
    if c == "dashboard":
        from dashboard import run_fastapi
        run_fastapi()
        return 0
    if c == "portfolio-backtest":
        from portfolio_backtest import main as m
        return m(rest)
    if c == "paper-report":
        from paper_report import main as m
        return m(rest)
    if c == "replay-report":
        from replay_report import main as m
        return m(rest)
    if c == "validate":
        from validation import main as m
        return m(rest)
    if c == "exchange-smoke":
        from exchange_smoke import main as m
        return m(rest)
    if c == "exchange-sandbox":
        from exchange_sandbox import main as m
        return m(rest)
    if c == "brain":
        print(json.dumps(BRAIN.snapshot(), indent=2))
        return 0
    if c == "ai-health":
        import multi_ai
        ok, msg = multi_ai.groq_healthcheck()
        print(msg)
        return 0 if ok else 1
    if c == "ai-test":
        import multi_ai
        ok, msg = multi_ai.groq_probe()
        print(msg)
        return 0 if ok else 1
    handlers = {"backtest": cmd_backtest, "learn": lambda a: cmd_backtest(a, learn=True), "optimize": cmd_optimize,
                "dca": cmd_dca, "grid": cmd_grid, "metrics": cmd_metrics, "status": cmd_status,
                "run": lambda a: cmd_run(a, live=False), "live": lambda a: cmd_run(a, live=True)}
    return handlers[c](args)


if __name__ == "__main__":
    raise SystemExit(main())
