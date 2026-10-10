import config
import db


def test_order_intent_is_idempotent_and_persistent(tmp_path, monkeypatch):
    monkeypatch.setattr(config.CFG, "db_path", str(tmp_path / "orders.db"))
    conn = db.connect()
    first = db.create_order_intent(conn, order_key="entry-1", client_order_id="zora-1", symbol="BTC/USDT", side="buy", order_type="market", amount=0.1, price=100)
    second = db.create_order_intent(conn, order_key="entry-1", client_order_id="zora-1", symbol="BTC/USDT", side="buy", order_type="market", amount=0.1, price=100)
    assert first["client_order_id"] == second["client_order_id"] == "zora-1"
    db.update_order_intent(conn, "entry-1", status="UNKNOWN", last_error="timeout")
    assert db.load_unresolved_orders(conn)[0]["status"] == "UNKNOWN"


def test_risk_reservation_is_idempotent_and_atomic_limit_enforced(tmp_path, monkeypatch):
    monkeypatch.setattr(config.CFG, "db_path", str(tmp_path / "risk.db"))
    conn = db.connect()
    assert db.reserve_risk(conn, "r1", "BTC/USDT", "live", 60, 100)
    assert db.reserve_risk(conn, "r1", "BTC/USDT", "live", 60, 100)
    assert not db.reserve_risk(conn, "r2", "ETH/USDT", "live", 50, 100)
    db.set_reservation_status(conn, "r1", "uncertain")
    assert not db.reserve_risk(conn, "r2", "ETH/USDT", "live", 50, 100)
