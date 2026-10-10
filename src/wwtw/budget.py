"""Thread- and process-safe API spend ledger with a hard cap (reserve before a call, settle after)."""

import csv
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from wwtw.config import Price


class BudgetExceeded(RuntimeError):
    """Raised before a paid call that would push total spend over the cap."""


@dataclass(frozen=True)
class Usage:
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int


def cost_usd(price: Price, usage: Usage) -> float:
    uncached = max(usage.input_tokens - usage.cached_input_tokens, 0)
    return (
        uncached * price.input
        + usage.cached_input_tokens * price.cached_input
        + usage.output_tokens * price.output
    ) / 1e6


_SCHEMA = """CREATE TABLE IF NOT EXISTS ledger (
    id TEXT PRIMARY KEY, ts REAL, model TEXT, namespace TEXT, status TEXT,
    input_tokens INTEGER, cached_input_tokens INTEGER, output_tokens INTEGER, cost_usd REAL)"""


class BudgetLedger:
    def __init__(self, db_path: Path, cap_usd: float, prices: dict[str, Price]):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.cap_usd = cap_usd
        self.prices = prices
        self._local = threading.local()
        self._conn().execute(_SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.db_path, timeout=60, isolation_level=None)
            conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn = conn
        return conn

    def _price(self, model: str) -> Price:
        if model not in self.prices:
            raise KeyError(
                f"No price configured for model {model!r}; add it to configs/experiment.yaml"
            )
        return self.prices[model]

    def spent(self) -> float:
        return float(
            self._conn().execute("SELECT COALESCE(SUM(cost_usd), 0) FROM ledger").fetchone()[0]
        )

    def reserve(
        self, model: str, max_input_tokens: int, max_output_tokens: int, namespace: str
    ) -> str:
        est = cost_usd(self._price(model), Usage(max_input_tokens, 0, max_output_tokens))
        conn = self._conn()
        conn.execute("BEGIN IMMEDIATE")
        try:
            total = conn.execute("SELECT COALESCE(SUM(cost_usd), 0) FROM ledger").fetchone()[0]
            if total + est > self.cap_usd:
                raise BudgetExceeded(
                    f"Call would bring spend to ${total + est:.4f} > cap ${self.cap_usd:.2f}"
                )
            rid = uuid.uuid4().hex
            conn.execute(
                "INSERT INTO ledger VALUES (?,?,?,?,?,?,?,?,?)",
                (rid, time.time(), model, namespace, "reserved", 0, 0, 0, est),
            )
        except BaseException:
            conn.execute("ROLLBACK")
            raise
        conn.execute("COMMIT")
        return rid

    def settle(self, reservation_id: str, usage: Usage) -> float:
        conn = self._conn()
        row = conn.execute("SELECT model FROM ledger WHERE id=?", (reservation_id,)).fetchone()
        if row is None:
            raise KeyError(f"unknown reservation {reservation_id}")
        cost = cost_usd(self._price(row[0]), usage)
        conn.execute(
            "UPDATE ledger SET status='settled', input_tokens=?, cached_input_tokens=?, "
            "output_tokens=?, cost_usd=? WHERE id=?",
            (
                usage.input_tokens,
                usage.cached_input_tokens,
                usage.output_tokens,
                cost,
                reservation_id,
            ),
        )
        return cost

    def release(self, reservation_id: str) -> None:
        self._conn().execute(
            "DELETE FROM ledger WHERE id=? AND status='reserved'", (reservation_id,)
        )

    def export_csv(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = (
            self._conn()
            .execute(
                "SELECT ts, model, namespace, status, input_tokens, cached_input_tokens, output_tokens, "
                "cost_usd FROM ledger WHERE status='settled' ORDER BY ts"
            )
            .fetchall()
        )
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "ts",
                    "model",
                    "namespace",
                    "status",
                    "input_tokens",
                    "cached_input_tokens",
                    "output_tokens",
                    "cost_usd",
                ]
            )
            w.writerows(rows)
