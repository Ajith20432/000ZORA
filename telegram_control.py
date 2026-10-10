"""Telegram alerts and remote control. Only messages from admin chat IDs are ever acted on."""
from __future__ import annotations

import html
import logging
from typing import Any

import requests

from config import CFG

BASE = "https://api.telegram.org/bot{}"
MAX_MESSAGE = 4000  # Telegram rejects >4096 chars; a long AI note used to make the whole alert silently vanish
logger = logging.getLogger(__name__)

COMMANDS = [
    ("start", "Bot status / welcome"), ("status", "Full bot status"), ("signal", "Latest signals"),
    ("btc", "BTC market context"), ("ai", "AI provider health"), ("brain", "Learning brain weights"),
    ("pnl", "Trading PnL summary"), ("positions", "Open positions"),
    ("metrics", "Win rate, profit factor, Sharpe, drawdown"), ("guards", "Safety guards and market context"),
    ("ai_usage", "AI API calls today and remaining quota"), ("pause_ai", "Pause AI; use technical indicators only"),
    ("resume_ai", "Resume rate-limited AI inference"), ("set_max_trades", "Set daily trade limit: /set_max_trades 5"),
    ("pause", "Pause new entries"), ("resume", "Resume entries"), ("kill", "Emergency kill switch"),
    ("help", "Command list"),
]


def enabled() -> bool:
    return bool(CFG.telegram_bot_token and CFG.telegram_chat_id)


def _admins() -> set[str]:
    ids = set(CFG.telegram_admin_chat_ids)
    if CFG.telegram_chat_id:
        ids.add(str(CFG.telegram_chat_id))
    return ids


def is_admin(chat_id: str | int) -> bool:
    return str(chat_id) in _admins()


def _api(method: str, **kwargs):
    if not enabled():
        return None
    return requests.post(BASE.format(CFG.telegram_bot_token) + f"/{method}", timeout=10, **kwargs)


def send(text: str, parse_mode: str | None = None) -> None:
    if not enabled():
        return
    if len(text) > MAX_MESSAGE:
        text = text[: MAX_MESSAGE - 1] + "…"
    try:
        payload = {"chat_id": CFG.telegram_chat_id, "text": text, "disable_web_page_preview": True}
        if parse_mode:
            payload["parse_mode"] = parse_mode
        r = _api("sendMessage", json=payload)
        if r is not None:
            r.raise_for_status()
    except Exception:
        logger.exception("Telegram sendMessage failed; trading loop will continue")



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
    def __init__(self) -> None:
        self.offset: int | None = None
        self.initialized = False

    def _get_updates(self) -> list:
        params = {"timeout": 0, "allowed_updates": ["message"]}
        if self.offset is not None:
            params["offset"] = self.offset
        r = requests.get(BASE.format(CFG.telegram_bot_token) + "/getUpdates", params=params, timeout=10)
        r.raise_for_status()
        return r.json().get("result", [])

    def drain(self) -> int:
        """
        Discard commands queued while the bot was offline. Without this a stale "/kill" or "/resume" sent
        yesterday would execute the moment the bot restarts. Returns how many were dropped.
        """
        if not enabled():
            return 0
        dropped = 0
        try:
            while True:
                updates = self._get_updates()
                if not updates:
                    break
                for u in updates:
                    self.offset = u.get("update_id", 0) + 1
                dropped += len(updates)
                if len(updates) < 100:
                    break
        except Exception:
            pass
        self.initialized = True
        return dropped

    def poll(self) -> list[str]:
        if not enabled():
            return []
        try:
            updates = self._get_updates()
        except Exception:
            return []
        commands = []
        for u in updates:
            self.offset = u.get("update_id", 0) + 1
            msg = u.get("message", {})
            chat_id = str(msg.get("chat", {}).get("id", ""))
            text = (msg.get("text") or "").strip()
            if text:
                self.initialized = True
            if not is_admin(chat_id) or not text.startswith("/"):
                continue
            head, *rest = text.split()
            commands.append(head.lower() + (" " + " ".join(rest) if rest else ""))
        return commands
