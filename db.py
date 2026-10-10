"""SQLite persistence for ZORA trading state and audit data."""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

from config import CFG

SCHEMA = {
    "trades": """CREATE TABLE IF NOT EXISTS trades (
        id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL NOT NULL, symbol TEXT NOT NULL, action TEXT NOT NULL,
        price REAL NOT NULL, size REAL NOT NULL, stop_loss REAL, take_profit REAL, mode TEXT NOT NULL,
        result TEXT NOT NULL DEFAULT 'open', pnl REAL, exchange_id TEXT DEFAULT 'htx')""",
    "equity_curve": """CREATE TABLE IF NOT EXISTS equity_curve (
        id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL NOT NULL, equity REAL NOT NULL)""",
    "decision_log": """CREATE TABLE IF NOT EXISTS decision_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL NOT NULL, symbol TEXT NOT NULL, action TEXT NOT NULL,
        confidence REAL NOT NULL, approved INTEGER NOT NULL, reason TEXT, llm_note TEXT)""",
    "order_journal": """CREATE TABLE IF NOT EXISTS order_journal (
        order_key TEXT PRIMARY KEY, client_order_id TEXT NOT NULL UNIQUE, exchange_id TEXT NOT NULL,
        symbol TEXT NOT NULL, side TEXT NOT NULL, order_type TEXT NOT NULL,
        requested_amount REAL NOT NULL, reference_price REAL, status TEXT NOT NULL,
        exchange_order_id TEXT, filled_amount REAL NOT NULL DEFAULT 0, average_fill_price REAL,
        fee_cost REAL, fee_currency TEXT, last_error TEXT, created_at REAL NOT NULL, updated_at REAL NOT NULL
    )""",
    "risk_reservations": """CREATE TABLE IF NOT EXISTS risk_reservations (
        reservation_id TEXT PRIMARY KEY, symbol TEXT NOT NULL, mode TEXT NOT NULL,
        reserved_risk REAL NOT NULL CHECK(reserved_risk >= 0), status TEXT NOT NULL,
        created_at REAL NOT NULL, updated_at REAL NOT NULL
    )""",
    "open_positions": """CREATE TABLE IF NOT EXISTS open_positions (
        symbol TEXT PRIMARY KEY, ts REAL NOT NULL, action TEXT NOT NULL, entry REAL NOT NULL, size REAL NOT NULL,
        stop_loss REAL, take_profit REAL, mode TEXT NOT NULL, protection_json TEXT DEFAULT '{}',
        expert_scores_json TEXT DEFAULT '{}', factor_scores_json TEXT DEFAULT '{}', initial_stop REAL)""",
}

_OPEN_POSITION_MIGRATIONS = {
    "expert_scores_json": "TEXT DEFAULT '{}'",
    "factor_scores_json": "TEXT DEFAULT '{}'",
    "initial_stop": "REAL",
}


def connect() -> sqlite3.Connection:
    path = Path(CFG.db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=10, check_same_thread=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=10000")
    for sql in SCHEMA.values():
        conn.execute(sql)
    # Migrate databases created by older versions.
    cols = {r[1] for r in conn.execute("PRAGMA table_info(open_positions)")}
    for col, typ in _OPEN_POSITION_MIGRATIONS.items():
        if col not in cols:
            conn.execute(f"ALTER TABLE open_positions ADD COLUMN {col} {typ}")
    trade_cols = {r[1] for r in conn.execute("PRAGMA table_info(trades)")}
    if "exchange_id" not in trade_cols:
        conn.execute("ALTER TABLE trades ADD COLUMN exchange_id TEXT DEFAULT 'htx'")
    conn.commit()
    return conn


def log_trade(conn, symbol, action, price, size, stop, take, mode, result="open", pnl=None, exchange_id=None):
    conn.execute(
        "INSERT INTO trades(ts,symbol,action,price,size,stop_loss,take_profit,mode,result,pnl,exchange_id) "
        "VALUES(?,?,?,?,?,?,?,?,?,?,?)",
        (time.time(), symbol, action, float(price), float(size), stop, take, mode, result, pnl, exchange_id or CFG.exchange_id))
    conn.commit()


def log_equity(conn, equity):
    conn.execute("INSERT INTO equity_curve(ts,equity) VALUES(?,?)", (time.time(), float(equity)))
    conn.commit()


def log_decision(conn, symbol, signal, approved, reason, llm_note=None):
    conn.execute(
        "INSERT INTO decision_log(ts,symbol,action,confidence,approved,reason,llm_note) VALUES(?,?,?,?,?,?,?)",
        (time.time(), symbol, signal.action, float(signal.confidence), int(bool(approved)), reason, llm_note))
    conn.commit()


def save_open_position(conn, p):
    conn.execute(
        """INSERT INTO open_positions(symbol,ts,action,entry,size,stop_loss,take_profit,mode,protection_json,
               expert_scores_json,factor_scores_json,initial_stop)
           VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
           ON CONFLICT(symbol) DO UPDATE SET ts=excluded.ts, action=excluded.action, entry=excluded.entry,
               size=excluded.size, stop_loss=excluded.stop_loss, take_profit=excluded.take_profit, mode=excluded.mode,
               protection_json=excluded.protection_json, expert_scores_json=excluded.expert_scores_json,
               factor_scores_json=excluded.factor_scores_json, initial_stop=excluded.initial_stop""",
        (p["symbol"], p.get("opened_ts", time.time()), p["action"], p["entry"], p["size"], p.get("stop"), p.get("take"),
         p.get("mode", "paper"), json.dumps(p.get("protection", {})), json.dumps(p.get("expert_scores", {})),
         json.dumps(p.get("factor_scores_by_tf", {})), p.get("initial_stop", p.get("stop"))))
    conn.commit()


def delete_open_position(conn, symbol):
    conn.execute("DELETE FROM open_positions WHERE symbol=?", (symbol,))
    conn.commit()


def load_open_positions(conn):
    rows = []
    for r in conn.execute("SELECT * FROM open_positions ORDER BY symbol"):
        d = dict(r)
        d["stop"] = d.pop("stop_loss")
        d["take"] = d.pop("take_profit")
        for col, key in (("protection_json", "protection"), ("expert_scores_json", "expert_scores"),
                         ("factor_scores_json", "factor_scores_by_tf")):
            try:
                d[key] = json.loads(d.pop(col) or "{}")
            except (TypeError, ValueError):
                d[key] = {}
        d["initial_stop"] = d.get("initial_stop") or d.get("stop")
        d["opened_ts"] = d.get("ts")
        rows.append(d)
    return rows


def summary(conn):
    r = conn.execute("SELECT COUNT(*) n, COALESCE(SUM(pnl),0) pnl FROM trades "
                     "WHERE result != 'open' AND pnl IS NOT NULL").fetchone()
    e = conn.execute("SELECT equity FROM equity_curve ORDER BY id DESC LIMIT 1").fetchone()
    return {"closed_trades": int(r["n"]), "realized_pnl": float(r["pnl"]),
            "latest_equity": float(e["equity"] if e else CFG.starting_capital)}


def peak_equity(conn):
    r = conn.execute("SELECT MAX(equity) FROM equity_curve").fetchone()
    return float(r[0]) if r and r[0] is not None else None


def recent_closed_pnls(conn, limit=500):
    rows = conn.execute("SELECT pnl FROM trades WHERE result != 'open' AND pnl IS NOT NULL "
                        "ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [float(r[0]) for r in reversed(rows)]


def equity_series(conn, limit=500):
    rows = conn.execute("SELECT ts,equity FROM equity_curve ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [(float(r[0]), float(r[1])) for r in reversed(rows)]


def recent_trades(conn, limit=20):
    rows = conn.execute("SELECT ts,symbol,action,price,size,result,pnl,exchange_id FROM trades "
                        "ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [dict(r) for r in rows][::-1]


def recent_decisions(conn, limit=15):
    rows = conn.execute("SELECT ts,symbol,action,confidence,approved,reason,llm_note FROM decision_log "
                        "ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [dict(r) for r in rows][::-1]


def create_order_intent(conn, *, order_key, client_order_id, symbol, side, order_type, amount, price, status="PERSISTED_INTENT"):
    """Persist an order intent before network submission; idempotent by logical key."""
    now = time.time()
    conn.execute("BEGIN IMMEDIATE")
    try:
        row = conn.execute("SELECT * FROM order_journal WHERE order_key=?", (order_key,)).fetchone()
        if row:
            if row["client_order_id"] != client_order_id:
                raise ValueError("order key already exists with a different client order ID")
            conn.commit()
            return dict(row)
        conn.execute(
            "INSERT INTO order_journal(order_key,client_order_id,exchange_id,symbol,side,order_type,requested_amount,reference_price,status,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            (order_key, client_order_id, CFG.exchange_id, symbol, side, order_type, float(amount), price, status, now, now),
        )
        conn.commit()
        return dict(conn.execute("SELECT * FROM order_journal WHERE order_key=?", (order_key,)).fetchone())
    except Exception:
        conn.rollback()
        raise

def update_order_intent(conn, order_key, *, status, exchange_order_id=None, filled_amount=None, average_fill_price=None, fee_cost=None, fee_currency=None, last_error=None):
    allowed = {"PERSISTED_INTENT", "SUBMITTING", "OPEN", "PARTIAL", "FILLED", "REJECTED", "CANCELED", "UNKNOWN", "RECONCILED"}
    if status not in allowed:
        raise ValueError(f"Invalid order status: {status}")
    fields = ["status=?", "updated_at=?"]
    values = [status, time.time()]
    for col, val in (("exchange_order_id", exchange_order_id), ("filled_amount", filled_amount),
                      ("average_fill_price", average_fill_price), ("fee_cost", fee_cost),
                      ("fee_currency", fee_currency), ("last_error", last_error)):
        if val is not None:
            fields.append(f"{col}=?"); values.append(val)
    values.append(order_key)
    cur = conn.execute(f"UPDATE order_journal SET {', '.join(fields)} WHERE order_key=?", values)
    conn.commit()
    if cur.rowcount == 0:
        raise KeyError(f"Unknown order intent: {order_key}")

def load_unresolved_orders(conn, mode=None):
    statuses = ("PERSISTED_INTENT", "SUBMITTING", "OPEN", "PARTIAL", "UNKNOWN")
    sql = "SELECT * FROM order_journal WHERE status IN (?,?,?,?,?)"
    args = list(statuses)
    if mode:
        sql += " AND exchange_id=?"; args.append(CFG.exchange_id)
    sql += " ORDER BY created_at"
    return [dict(r) for r in conn.execute(sql, args).fetchall()]

def reserve_risk(conn, reservation_id, symbol, mode, risk_amount, max_risk_amount):
    """Atomic, idempotent risk reservation across connections/processes."""
    now = time.time(); risk_amount = float(risk_amount); max_risk_amount = float(max_risk_amount)
    if risk_amount <= 0 or max_risk_amount <= 0:
        return False
    conn.execute("BEGIN IMMEDIATE")
    try:
        old = conn.execute("SELECT status FROM risk_reservations WHERE reservation_id=?", (reservation_id,)).fetchone()
        if old:
            conn.commit(); return old["status"] in ("reserved", "committed", "uncertain")
        active = conn.execute("SELECT COALESCE(SUM(reserved_risk),0) FROM risk_reservations WHERE mode=? AND status IN ('reserved','committed','uncertain')", (mode,)).fetchone()[0]
        if float(active) + risk_amount > max_risk_amount:
            conn.rollback(); return False
        conn.execute("INSERT INTO risk_reservations VALUES(?,?,?,?,?,?,?)", (reservation_id, symbol, mode, risk_amount, "reserved", now, now))
        conn.commit(); return True
    except Exception:
        conn.rollback(); raise

def set_reservation_status(conn, reservation_id, status):
    if status not in {"reserved", "committed", "released", "uncertain"}:
        raise ValueError(f"Invalid reservation status: {status}")
    cur = conn.execute("UPDATE risk_reservations SET status=?,updated_at=? WHERE reservation_id=?", (status, time.time(), reservation_id))
    conn.commit()
    if cur.rowcount == 0:
        raise KeyError(f"Unknown risk reservation: {reservation_id}")
