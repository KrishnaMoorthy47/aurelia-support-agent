"""Mock back-office data in SQLite."""
from __future__ import annotations

import json
import os
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

SEED_CUSTOMERS = [
    # id, name, account_type, base_ccy, status, kyc_tier, kyc_status, missing_docs, daily_limit, withdrawals_this_month, open_positions
    ("C-1001", "Aisha Rahman", "Standard", "USD", "active", 2, "verified", [], 10000, 1, 1),
    ("C-1002", "Daniel Okafor", "Raw", "EUR", "active", 1, "pending_documents", ["proof_of_address"], 2000, 0, 0),
    ("C-1003", "Priya Nair", "Standard", "AED", "restricted", 2, "verified", [], 10000, 0, 0),
    ("C-1004", "Marcus Lee", "Islamic (Standard)", "USD", "active", 1, "verified", [], 2000, 2, 0),
]

TIER_LIMITS = {1: 2000, 2: 10000, 3: 50000}


def _db_path() -> str:
    return os.environ.get("MOCK_DB_PATH", "data/aurelia_mock.db")


def connect(path: str | None = None) -> sqlite3.Connection:
    path = path or _db_path()
    if path != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        create table if not exists customers (
          id text primary key, name text, account_type text, base_ccy text, status text,
          kyc_tier integer, kyc_status text, missing_docs text, daily_limit integer,
          withdrawals_this_month integer, open_positions integer
        );
        create table if not exists tickets (
          id integer primary key autoincrement, customer_id text, category text, summary text,
          priority text, status text, created_at text
        );
        create table if not exists limit_requests (
          id integer primary key autoincrement, customer_id text, requested_limit integer,
          reason text, approved_by text, created_at text
        );
        create table if not exists closure_requests (
          id integer primary key autoincrement, customer_id text, reason text, approved_by text, created_at text
        );
        """
    )
    if conn.execute("select count(*) from customers").fetchone()[0] == 0:
        for c in SEED_CUSTOMERS:
            conn.execute(
                "insert into customers values (?,?,?,?,?,?,?,?,?,?,?)",
                (*c[:7], json.dumps(c[7]), *c[8:]),
            )
        conn.commit()
    return conn


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def get_customer(conn: sqlite3.Connection, customer_id: str) -> sqlite3.Row | None:
    return conn.execute("select * from customers where id = ?", (customer_id,)).fetchone()


def reset(path: str | None = None) -> None:
    """Put the mock back office back to its seed state (used by the eval runner and the demo's reset button)."""
    with closing(connect(path)) as conn:
        conn.executescript(
            "delete from customers; delete from tickets; delete from limit_requests; delete from closure_requests;"
            "delete from sqlite_sequence;"
        )
        conn.commit()
    connect(path).close()  # reseeds the customers
