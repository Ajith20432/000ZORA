"""Safe execution adapters with native protection and reconciliation helpers."""
from __future__ import annotations

import time
import hashlib
import logging

logger = logging.getLogger(__name__)

try:
    import ccxt
except ImportError:  # keep paper/replay/contract tests usable without the optional exchange client
    class _CCXTNetworkError(Exception):
        pass

    class _CCXTBaseError(Exception):
        pass

    class _CCXTExchangeError(_CCXTBaseError):
        pass

    class _CCXTFallback:
        NetworkError = _CCXTNetworkError
        BaseError = _CCXTBaseError
        ExchangeError = _CCXTExchangeError

    ccxt = _CCXTFallback()

from config import CFG


class PaperExecutor:
    live = False
    last_amount = None

    def open(self, exchange, symbol, action, amount, price):
        return price

    def close(self, exchange, symbol, action, amount, price):
        return price

    def place_protection(self, *args, **kwargs):
        return {"stop": "paper", "take": "paper"}

    def cancel_protection(self, *args, **kwargs):
        return True


class LiveExecutor:
    live = True

    def __init__(self, telegram_alert=lambda msg: None):
        self.alert = telegram_alert
        self.last_amount = None     # exchange-precision amount of the most recent order actually sent
        self.last_outcome = "NONE"
        self.last_order = None

    def _sized_amount(self, exchange, symbol, amount):
        try:
            market = exchange.market(symbol)
            amount = float(exchange.amount_to_precision(symbol, amount))
            min_amount = ((market.get("limits", {}) or {}).get("amount", {}) or {}).get("min")
            if min_amount and amount < min_amount:
                return None
            return amount
        except Exception:
            return amount

    @staticmethod
    def _market_buy_needs_price(exchange, side: str) -> bool:
        """HTX (and a few others) price a MARKET BUY by cost = amount * price, so ccxt refuses it without a price."""
        opts = getattr(exchange, "options", None) or {}
        return side == "buy" and bool(opts.get("createMarketBuyOrderRequiresPrice", True))

    @staticmethod
    def client_order_id(order_key: str) -> str:
        # Stable for one logical order; never change this ID during reconciliation.
        digest = hashlib.sha256(str(order_key).encode("utf-8")).hexdigest()[:24]
        return f"zora-{digest}"

    def _find_by_client_id(self, exchange, symbol, client_id):
        # CCXT exchange adapters vary. Never assume a missing lookup means no order.
        for method_name in ("fetch_order_by_client_order_id", "fetch_order_by_client_id"):
            method = getattr(exchange, method_name, None)
            if callable(method):
                try:
                    found = method(client_id, symbol)
                    if found:
                        return found
                except Exception as exc:
                    logger.warning("client-order lookup failed for %s: %s", client_id, exc)
        return None

    def _place(self, exchange, symbol, side, amount, ref_price=None, order_key=None):
        amount = self._sized_amount(exchange, symbol, amount)
        self.last_amount = amount
        self.last_outcome, self.last_order = "NONE", None
        if not amount:
            self.last_outcome = "REJECTED"
            self.alert(f"[live] {symbol} {side} skipped — below the exchange minimum")
            return None
        logical_key = order_key or f"{getattr(exchange, 'id', 'exchange')}:{symbol}:{side}:{time.time_ns()}"
        client_id = self.client_order_id(logical_key)
        params = {"clientOrderId": client_id, "clientOrderID": client_id}
        order_price = ref_price if (ref_price and self._market_buy_needs_price(exchange, side)) else None
        self.last_outcome = "SUBMITTING"
        try:
            order = exchange.create_order(symbol, "market", side, amount, order_price, params)
            self.last_order = order
            self.last_outcome = "FILLED" if str(order.get("status", "")).lower() in ("closed", "filled") else "OPEN"
            return order
        except ccxt.NetworkError as exc:
            logger.error("order submission outcome uncertain for %s (%s): %s", client_id, symbol, exc)
            found = self._find_by_client_id(exchange, symbol, client_id)
            if found:
                self.last_order = found
                self.last_outcome = "FILLED" if str(found.get("status", "")).lower() in ("closed", "filled") else "OPEN"
                return found
            # Crucial: no blind retry after a timeout. The exchange may have accepted it.
            self.last_outcome = "UNKNOWN"
            self.alert(f"[live] CRITICAL {symbol} {side}: outcome UNKNOWN for client order {client_id}; do not resubmit until reconciled")
            return None
        except ccxt.BaseError as exc:
            self.last_outcome = "REJECTED"
            logger.warning("exchange rejected order %s: %s", client_id, exc)
            self.alert(f"[live] {symbol} {side} rejected: {exc}")
            return None
        except Exception as exc:
            # Unknown adapter errors after create_order begins are conservatively ambiguous.
            self.last_outcome = "UNKNOWN"
            logger.exception("unexpected order submission error for %s", client_id)
            self.alert(f"[live] CRITICAL {symbol} {side}: uncertain order result ({exc}); reconcile before retry")
            return None

    def open(self, exchange, symbol, action, amount, price, order_key=None):
        side = "buy" if action == "BUY" else "sell"
        order = self._place(exchange, symbol, side, amount, price, order_key=order_key)
        if order is None:
            return None
        if self.last_outcome != "FILLED":
            self.alert(f"[live] {symbol} entry is {self.last_outcome}; fill/protection reconciliation required")
            return None
        fill = order.get("average") or order.get("price") or price
        self.alert(f"[live] OPENED {action} {symbol} amount={self.last_amount or amount} @ ~{fill} (order state: {self.last_outcome})")
        return float(fill)

    def close(self, exchange, symbol, action, amount, price, order_key=None):
        side = "sell" if action == "BUY" else "buy"
        order = self._place(exchange, symbol, side, amount, price, order_key=order_key)
        if order is None:
            return None
        fill = order.get("average") or order.get("price") or price
        self.alert(f"[live] EXIT ORDER {self.last_outcome} {action} {symbol} amount={self.last_amount or amount} @ ~{fill}")
        return float(fill)

    def _native_supported(self, exchange, symbol: str) -> bool:
        try:
            if hasattr(exchange, "featureValue"):
                return bool(exchange.featureValue(symbol, "createOrder", "stopLossPrice") or
                            exchange.featureValue(symbol, "createOrder", "takeProfitPrice"))
        except Exception:
            pass
        # Never assume support: protection is safety-critical.
        return False

    def place_protection(self, exchange, symbol, action, amount, stop, take):
        """Place exchange-native conditional exits. Refuse to trade unprotected if required."""
        if not self._native_supported(exchange, symbol):
            msg = f"[live] {symbol} native stop/take-profit capability not confirmed by exchange adapter"
            self.alert(msg)
            if CFG.native_protection_required:
                return None
            return {"software_only": True}
        exit_side = "sell" if action == "BUY" else "buy"
        base = {}
        try:
            # reduceOnly is a derivatives flag; spot markets reject it. Unknown market -> keep it (safer on swaps).
            if exchange.market(symbol).get("spot"):
                base = {}
            else:
                base = {"reduceOnly": True}
        except Exception:
            base = {"reduceOnly": True}
        orders = {}
        ex_id = getattr(exchange, "id", "ex")
        stamp = int(time.time() * 1000)
        try:
            orders["stop"] = exchange.create_order(
                symbol, "market", exit_side, amount, None,
                {"stopLossPrice": stop, **base, "clientOrderId": f"zora-sl-{ex_id}-{stamp}"})
            orders["take"] = exchange.create_order(
                symbol, "market", exit_side, amount, None,
                {"takeProfitPrice": take, **base, "clientOrderId": f"zora-tp-{ex_id}-{stamp}"})
            self.alert(f"[live] NATIVE PROTECTION armed {symbol}: SL={stop} TP={take}")
            return orders
        except ccxt.BaseError as e:
            self.alert(f"[live] native protection FAILED for {symbol}: {e}")
            # Don't leave half a bracket behind: an orphaned stop would later fire on its own
            # and open an unwanted position in the opposite direction.
            if orders:
                self.cancel_protection(exchange, symbol, orders)
            return None

    def protection_triggered(self, exchange, symbol, protection):
        """
        Did the exchange already close the position via its native stop/take order?
        Returns ("stop"|"take", average_fill_price_or_None) or None. Without this, the bot's own
        price check would fire a SECOND closing order on top of the one the exchange already filled.
        """
        if not protection or protection.get("software_only"):
            return None
        for key in ("stop", "take"):
            oid = (protection.get(key) or {}).get("id")
            if not oid:
                continue
            try:
                order = exchange.fetch_order(oid, symbol)
            except Exception:
                continue
            if str(order.get("status", "")).lower() in ("closed", "filled"):
                return key, (order.get("average") or order.get("price"))
        return None

    def cancel_protection(self, exchange, symbol, protection):
        if not protection or protection.get("software_only"):
            return True
        ok = True
        for key in ("stop", "take"):
            order = protection.get(key) or {}
            oid = order.get("id")
            if not oid:
                continue
            try:
                exchange.cancel_order(oid, symbol)
            except ccxt.BaseError as e:
                ok = False
                self.alert(f"[live] failed to cancel {key} protection {oid}: {e}")
        return ok

    def reconcile_order(self, exchange, intent):
        """Query a persisted order by exchange ID/client ID without ever resubmitting it."""
        symbol = intent["symbol"]
        client_id = intent["client_order_id"]
        order = None
        exchange_id = intent.get("exchange_order_id")
        if exchange_id:
            try:
                order = exchange.fetch_order(exchange_id, symbol)
            except Exception as exc:
                logger.warning("fetch_order(%s) failed during reconciliation: %s", exchange_id, exc)
        if order is None:
            order = self._find_by_client_id(exchange, symbol, client_id)
        if order is None:
            # Search recent trades for an exchange-supported client order ID where available.
            try:
                trades = exchange.fetch_my_trades(symbol, limit=100)
                order = next((t for t in trades if str((t.get("info") or {}).get("clientOrderId") or
                                                       (t.get("info") or {}).get("clientOrderID") or
                                                       t.get("clientOrderId") or "") == client_id), None)
            except Exception as exc:
                logger.warning("recent-trade reconciliation unavailable for %s: %s", client_id, exc)
        if order is None:
            return {"state": "UNKNOWN", "reason": "exchange cannot authoritatively locate the order", "order": None}
        status = str(order.get("status", "")).lower()
        if status in ("closed", "filled"):
            state = "FILLED"
        elif status in ("canceled", "cancelled", "rejected", "expired"):
            state = "REJECTED" if status == "rejected" else "CANCELED"
        elif status in ("open", "new", "partially_filled"):
            state = "PARTIAL" if float(order.get("filled") or 0) > 0 else "OPEN"
        else:
            state = "UNKNOWN"
        return {"state": state, "reason": "located by exchange reconciliation", "order": order}

    def reconcile(self, exchange, symbol, expected_position=None):
        """Return exchange truth. If it cannot be proven, return safe=False (callers must then refuse entries)."""
        try:
            open_orders = exchange.fetch_open_orders(symbol)
        except Exception as e:
            return {"safe": False, "reason": f"cannot fetch open orders: {e}", "open_orders": []}
        result = {"safe": True, "open_orders": open_orders, "position": None}
        try:
            try:
                is_spot = bool(exchange.market(symbol).get("spot"))
            except Exception:
                is_spot = False
            if not is_spot and exchange.has.get("fetchPositions"):
                positions = exchange.fetch_positions([symbol])
                result["position"] = next(
                    (p for p in positions if abs(float(p.get("contracts") or p.get("contractSize") or 0)) > 0), None)
            elif exchange.has.get("fetchBalance"):
                base = symbol.split("/")[0]
                balance = exchange.fetch_balance()
                result["position"] = {"contracts": float((balance.get("total") or {}).get(base) or 0), "spot_balance": True}
        except Exception as e:
            return {"safe": False, "reason": f"cannot reconcile position: {e}", "open_orders": open_orders}
        return result
