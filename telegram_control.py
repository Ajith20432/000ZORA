"""Telegram alerts and remote control. Only messages from admin chat IDs are ever acted on."""
from __future__ import annotations

import html
import logging
import random
import threading
import time
from collections import defaultdict, deque
from typing import Any

import requests

from config import CFG

BASE = "https://api.telegram.org/bot{}"
MAX_MESSAGE = 3900  # keep margin below Telegram's 4096-character limit
POLL_TIMEOUT_SECONDS = 20
REQUEST_TIMEOUT_SECONDS = 25
MAX_UPDATES_PER_POLL = 100
MAX_COMMANDS_PER_MINUTE = 20
logger = logging.getLogger(__name__)
_send_lock = threading.Lock()

COMMANDS = [
    ("start", "Bot status / welcome"), ("status", "Full bot status"), ("signal", "Latest signals"),
    ("btc", "BTC market context"), ("ai", "AI provider health"), ("brain", "Learning brain weights"),
    ("pnl", "Trading PnL summary"), ("positions", "Open positions"),
    ("metrics", "Win rate, profit factor, Sharpe, drawdown"), ("guards", "Safety guards and market context"),
    ("ai_usage", "AI API calls today and remaining quota"), ("pause_ai", "Pause AI; use technical indicators only"),
    ("resume_ai", "Resume rate-limited AI inference"), ("set_max_trades", "Set daily trade limit: /set_max_trades 5"),
    ("pause", "Pause new entries"), ("resume", "Resume entries"), ("kill", "Emergency kill switch"),
    ("trades", "Open trades with prices and PnL"), ("ai_votes", "Latest AI advisory votes"),
    ("start_bot", "Resume automated trading entries"), ("stop_bot", "Pause automated trading entries"),
    ("help", "Command list"),
]


def enabled() -> bool:
    return bool(CFG.telegram_bot_token and CFG.telegram_chat_id)


def _admins() -> set[str]:
    # Admin commands are authorized only by the explicit allow-list.
    return set(CFG.telegram_admin_chat_ids)


def is_admin(chat_id: str | int) -> bool:
    return str(chat_id) in _admins()


def _api(method: str, **kwargs):
    if not enabled():
        return None
    return requests.post(BASE.format(CFG.telegram_bot_token) + f"/{method}", timeout=10, **kwargs)


def _chunks(text: str, limit: int = MAX_MESSAGE) -> list[str]:
    """Split long Telegram text at line boundaries where possible."""
    if len(text) <= limit:
        return [text]
    chunks: list[str] = []
    current = ""
    for line in text.splitlines(keepends=True):
        while len(line) > limit:
            if current:
                chunks.append(current)
                current = ""
            chunks.append(line[:limit])
            line = line[limit:]
        if len(current) + len(line) > limit:
            if current:
                chunks.append(current)
            current = line
        else:
            current += line
    if current:
        chunks.append(current)
    return chunks or [text[:limit]]


def send(text: str, parse_mode: str | None = None) -> None:
    """Best-effort alert delivery; network/API errors never interrupt trading.

    Long alerts are split rather than silently truncated. HTML callers should
    keep tags balanced per chunk; trade alerts are deliberately kept short.
    """
    if not enabled():
        return
    try:
        with _send_lock:
            for part in _chunks(str(text)):
                payload = {"chat_id": CFG.telegram_chat_id, "text": part,
                           "disable_web_page_preview": True}
                if parse_mode:
                    payload["parse_mode"] = parse_mode
                r = _api("sendMessage", json=payload)
                if r is not None:
                    r.raise_for_status()
                    data = r.json()
                    if isinstance(data, dict) and data.get("ok") is False:
                        raise RuntimeError(f"Telegram API rejected sendMessage: {data.get('description', 'unknown error')}")
    except Exception:
        logger.exception("Telegram sendMessage failed; trading loop will continue")


def send_health_alert(title: str, details: str, *, severity: str = "INFO") -> None:
    """Send a concise, HTML-safe operational alert."""
    sev = str(severity).upper()
    icon = {"CRITICAL": "🚨", "ERROR": "🔴", "WARNING": "🟠", "INFO": "ℹ️"}.get(sev, "ℹ️")
    body = f"{icon} <b>ZORA {html.escape(sev)}</b>\n<b>{html.escape(str(title))}</b>\n{html.escape(str(details))[:2500]}"
    send(body, parse_mode="HTML")


def send_heartbeat(status: str, *, equity: float | None = None,
                   open_positions: int | None = None, mode: str = "PAPER") -> None:
    """Optional compact heartbeat; caller controls the configured interval."""
    fields = [f"Mode: {html.escape(str(mode))}", f"Status: {html.escape(str(status))}"]
    if equity is not None:
        fields.append(f"Equity: {float(equity):.2f}")
    if open_positions is not None:
        fields.append(f"Open positions: {int(open_positions)}")
    send("💓 <b>ZORA HEARTBEAT</b>\n" + "\n".join(fields), parse_mode="HTML")



def trade_alert(
    *,
    symbol: str,
    action: str,
    mode: str,
    quantity: float,
    fill_price: float,
    stop_loss: float | None,
    take_profit: float | None,
    decision_model: str,
    strategy_confidence: float,
    ai_consensus: float | None,
    providers: list[str] | None = None,
    rationale: str = "No AI rationale supplied; see strategy factors.",
) -> None:
    """Send a consistently formatted, HTML-safe trade decision alert.

    Telegram's HTML parse mode is used because it allows reliable escaping of
    arbitrary provider/rationale text while still supporting bold headings.
    Scores are labelled explicitly to avoid confusing strategy confidence with
    the directional strength of the multi-provider consensus.
    """
    normalized_action = str(action).upper()
    if normalized_action not in {"BUY", "SELL"}:
        logger.warning("Refusing to format invalid trade action: %r", action)
        return

    safe = lambda value: html.escape(str(value), quote=True)
    icon = "🟢" if normalized_action == "BUY" else "🔴"
    heading = "BUY ORDER / LONG ENTRY" if normalized_action == "BUY" else "SELL SIGNAL / SHORT ENTRY"
    provider_text = ", ".join(dict.fromkeys(providers or [])) or "Not identified by provider metadata"
    if ai_consensus is None:
        consensus_line = "<b>AI consensus strength:</b> N/A (AI unavailable or not used)"
    else:
        consensus_line = f"<b>AI consensus strength:</b> {abs(float(ai_consensus)):.1f}% ({'bullish' if ai_consensus > 0 else 'bearish' if ai_consensus < 0 else 'neutral'} direction)"

    stop_text = f"{float(stop_loss):.8g}" if stop_loss is not None else "N/A"
    take_text = f"{float(take_profit):.8g}" if take_profit is not None else "N/A"
    body = (
        f"{icon} <b>ZORA TRADE DECISION</b> {icon}\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"<b>{safe(heading)}</b>\n"
        f"<b>Symbol:</b> {safe(symbol)}\n"
        f"<b>Mode:</b> {safe(mode)}\n"
        f"<b>Decision model:</b> {safe(decision_model)}\n"
        f"<b>AI provider(s):</b> {safe(provider_text)}\n"
        f"<b>Strategy confidence:</b> {float(strategy_confidence):.1f}%\n"
        f"{consensus_line}\n"
        f"<b>Quantity:</b> {float(quantity):.8g}\n"
        f"<b>Fill price:</b> {float(fill_price):.8g}\n"
        f"<b>Stop loss:</b> {stop_text}  |  <b>Take profit:</b> {take_text}\n"
        f"<b>AI rationale:</b> {safe((rationale or 'No rationale supplied')[:900])}\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"<i>Automated decision alert — verify exchange fills and protection.</i>"
    )
    send(body, parse_mode="HTML")


def setup_commands() -> None:
    """Show the command menu in Telegram."""
    try:
        _api("setMyCommands", json={"commands": [{"command": c, "description": d} for c, d in COMMANDS]})
    except Exception:
        pass


class CommandPoller:
    """Safe long-poll command reader for the legacy ZORA loop.

    Configure TELEGRAM_ADMIN_CHAT_IDS with numeric Telegram user IDs. In a
    private chat, old installations that provide only chat.id remain supported;
    in group chats, authorization checks the actual sender's from.id.
    """
    def __init__(self) -> None:
        self.offset: int | None = None
        self.initialized = False
        self._seen: set[int] = set()
        self._seen_order: deque[int] = deque(maxlen=2000)
        self._rate: dict[str, deque[float]] = defaultdict(deque)
        self._last_error_log = 0.0

    def _get_updates(self) -> list:
        params = {"timeout": POLL_TIMEOUT_SECONDS,
                  "allowed_updates": '["message"]'}
        if self.offset is not None:
            params["offset"] = self.offset
        r = requests.get(BASE.format(CFG.telegram_bot_token) + "/getUpdates",
                         params=params, timeout=REQUEST_TIMEOUT_SECONDS)
        r.raise_for_status()
        data = r.json()
        if not isinstance(data, dict) or data.get("ok") is False:
            raise RuntimeError("Telegram getUpdates returned an unsuccessful response")
        result = data.get("result", [])
        return result if isinstance(result, list) else []

    def _authorized(self, message: dict) -> bool:
        chat = message.get("chat") or {}
        sender = message.get("from") or {}
        chat_id = str(chat.get("id", ""))
        sender_id = str(sender.get("id", ""))
        chat_type = str(chat.get("type", "private"))
        # In groups, never trust chat.id as an admin identity: that would grant
        # every member of the group control over the bot.
        if chat_type != "private":
            return bool(sender_id and is_admin(sender_id))
        return is_admin(sender_id or chat_id)

    def _within_rate_limit(self, identity: str) -> bool:
        now = time.monotonic()
        q = self._rate[identity]
        while q and now - q[0] >= 60:
            q.popleft()
        if len(q) >= MAX_COMMANDS_PER_MINUTE:
            return False
        q.append(now)
        return True

    def drain(self) -> int:
        """Discard stale queued updates at boot to avoid executing old commands."""
        if not enabled():
            return 0
        dropped = 0
        try:
            while True:
                updates = self._get_updates()
                if not updates:
                    break
                for u in updates:
                    self.offset = max(self.offset or 0, int(u.get("update_id", 0)) + 1)
                dropped += len(updates)
                if len(updates) < MAX_UPDATES_PER_POLL:
                    break
        except Exception:
            logger.exception("Telegram update drain failed")
        self.initialized = True
        return dropped

    def poll(self) -> list[str]:
        if not enabled():
            return []
        try:
            updates = self._get_updates()
        except Exception as exc:
            # Avoid log spam if Telegram/network remains unavailable.
            now = time.monotonic()
            if now - self._last_error_log > 60:
                logger.warning("Telegram polling unavailable: %s", exc)
                self._last_error_log = now
            return []
        commands: list[str] = []
        for update in updates[:MAX_UPDATES_PER_POLL]:
            try:
                update_id = int(update.get("update_id", -1))
            except (TypeError, ValueError):
                continue
            self.offset = max(self.offset or 0, update_id + 1)
            if update_id >= 0 and update_id in self._seen:
                continue
            if update_id >= 0:
                if len(self._seen_order) == self._seen_order.maxlen:
                    old = self._seen_order.popleft()
                    self._seen.discard(old)
                self._seen.add(update_id)
                self._seen_order.append(update_id)
            msg = update.get("message") or {}
            text = (msg.get("text") or "").strip()
            if not text.startswith("/") or not self._authorized(msg):
                continue
            identity = str((msg.get("from") or {}).get("id") or (msg.get("chat") or {}).get("id", ""))
            if not self._within_rate_limit(identity):
                logger.warning("Telegram command rate limit reached for admin %s", identity)
                continue
            head, *rest = text.split()
            # Telegram group commands may be addressed as /status@BotName.
            head = head.split("@", 1)[0].lower()
            command = head + (" " + " ".join(rest) if rest else "")
            commands.append(command)
        return commands
