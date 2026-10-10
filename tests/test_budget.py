import csv
import threading

import pytest

from wwtw.budget import BudgetExceeded, BudgetLedger, Usage, cost_usd
from wwtw.config import Price

PRICES = {"m": Price(input=1.0, cached_input=0.5, output=2.0)}  # USD per 1M tokens


def test_cost_formula():
    u = Usage(input_tokens=1_000_000, cached_input_tokens=400_000, output_tokens=500_000)
    assert cost_usd(PRICES["m"], u) == pytest.approx(0.6 * 1.0 + 0.4 * 0.5 + 0.5 * 2.0)


def test_reserve_settle_release(tmp_path):
    led = BudgetLedger(tmp_path / "l.sqlite", cap_usd=1.0, prices=PRICES)
    rid = led.reserve("m", 100_000, 100_000, "main")  # est 0.1 + 0.2 = 0.3
    assert led.spent() == pytest.approx(0.3)
    led.settle(rid, Usage(50_000, 0, 10_000))  # 0.05 + 0.02
    assert led.spent() == pytest.approx(0.07)
    rid2 = led.reserve("m", 100_000, 0, "main")
    led.release(rid2)
    assert led.spent() == pytest.approx(0.07)


def test_cap_blocks_and_records_nothing(tmp_path):
    led = BudgetLedger(tmp_path / "l.sqlite", cap_usd=0.25, prices=PRICES)
    with pytest.raises(BudgetExceeded):
        led.reserve("m", 100_000, 100_000, "main")  # 0.3 > 0.25
    assert led.spent() == 0.0


def test_unknown_model(tmp_path):
    led = BudgetLedger(tmp_path / "l.sqlite", cap_usd=1.0, prices=PRICES)
    with pytest.raises(KeyError, match="price"):
        led.reserve("nope", 1, 1, "main")


def test_persists_across_instances(tmp_path):
    p = tmp_path / "l.sqlite"
    led = BudgetLedger(p, cap_usd=1.0, prices=PRICES)
    led.settle(led.reserve("m", 0, 100_000, "main"), Usage(0, 0, 100_000))
    assert BudgetLedger(p, cap_usd=1.0, prices=PRICES).spent() == pytest.approx(0.2)


def test_concurrent_reservations_respect_cap(tmp_path):
    led = BudgetLedger(tmp_path / "l.sqlite", cap_usd=1.0, prices=PRICES)
    ok, blocked = [], []

    def worker():
        try:
            ok.append(led.reserve("m", 0, 100_000, "main"))  # 0.2 each -> only 5 fit
        except BudgetExceeded:
            blocked.append(1)

    threads = [threading.Thread(target=worker) for _ in range(12)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(ok) == 5 and len(blocked) == 7


def test_export_csv(tmp_path):
    led = BudgetLedger(tmp_path / "l.sqlite", cap_usd=1.0, prices=PRICES)
    led.settle(led.reserve("m", 10, 10, "main"), Usage(10, 0, 10))
    out = tmp_path / "ledger.csv"
    led.export_csv(out)
    with open(out) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1 and rows[0]["model"] == "m" and rows[0]["status"] == "settled"
