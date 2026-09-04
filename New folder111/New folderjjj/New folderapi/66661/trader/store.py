"""SQLite candle store: incremental bootstrap + append. Local-first, zero-cost."""
from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)


class StoreError(RuntimeError):
    """The candle store file is corrupt or unusable."""
    RECOVERY = (
        "The candle store is corrupt. Recovery: delete or move the file "
        "(it will be rebuilt from the free API on the next run), e.g.\n"
        "  mv <db_path> <db_path>.corrupt\n"
        "then re-run the command.")

SCHEMA = """
CREATE TABLE IF NOT EXISTS candles (
    instrument TEXT NOT NULL,
    granularity TEXT NOT NULL,
    time TEXT NOT NULL,
    open REAL NOT NULL, high REAL NOT NULL, low REAL NOT NULL, close REAL NOT NULL,
    volume INTEGER NOT NULL,
    PRIMARY KEY (instrument, granularity, time)
);
CREATE TABLE IF NOT EXISTS signals_log (
    ts TEXT NOT NULL, instrument TEXT NOT NULL, action TEXT NOT NULL,
    score REAL, regime TEXT, price REAL, units INTEGER, sl REAL, tp REAL,
    note TEXT
);
CREATE INDEX IF NOT EXISTS idx_signals_inst_note ON signals_log(instrument, note);
"""


class CandleStore:
    def __init__(self, db_path: str):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        try:
            self.conn = sqlite3.connect(db_path)
            self.conn.executescript(SCHEMA)
            self.conn.commit()
        except sqlite3.DatabaseError as e:
            raise StoreError(f"{db_path}: {e}\n{StoreError.RECOVERY}") from e

    def upsert(self, df: pd.DataFrame, instrument: str, granularity: str) -> int:
        if df.empty:
            return 0
        rows = [
            (instrument, granularity, t.strftime("%Y-%m-%dT%H:%M:%S.000000000Z"),
             float(o), float(h), float(l), float(c), int(v))
            for t, o, h, l, c, v in zip(df["time"], df["open"], df["high"],
                                        df["low"], df["close"], df["volume"])
        ]
        self.conn.executemany(
            "INSERT OR REPLACE INTO candles VALUES (?,?,?,?,?,?,?,?)", rows)
        self.conn.commit()
        return len(rows)

    def load(self, instrument: str, granularity: str) -> pd.DataFrame:
        q = ("SELECT time, open, high, low, close, volume FROM candles "
             "WHERE instrument=? AND granularity=? ORDER BY time")
        df = pd.read_sql_query(q, self.conn, params=(instrument, granularity))
        if df.empty:
            return df
        df["time"] = pd.to_datetime(df["time"], utc=True)
        return df.reset_index(drop=True)

    def log_signal(self, **kw) -> None:
        self.conn.execute(
            "INSERT INTO signals_log (ts, instrument, action, score, regime, price, units, sl, tp, note) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            (kw.get("ts"), kw.get("instrument"), kw.get("action"), kw.get("score"),
             kw.get("regime"), kw.get("price"), kw.get("units"), kw.get("sl"),
             kw.get("tp"), kw.get("note")))
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()
