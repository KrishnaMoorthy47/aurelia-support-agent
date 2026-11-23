"""Back-office MCP server (stdio): python -m app.mcp_server.server

The write tools refuse to run without `approved_by`, which the graph only passes after a reviewer approves.
"""
from __future__ import annotations

import json

from mcp.server.mcpserver import MCPServer

from app.mcp_server import db

mcp = MCPServer("aurelia-backoffice")


def _customer_or_error(conn, customer_id: str):
    row = db.get_customer(conn, customer_id)
    return row, (None if row else {"error": "customer_not_found"})


@mcp.tool()
def get_account_summary(customer_id: str) -> dict:
    """Overview of the customer's own account: type, status, verification tier, open tickets."""
    with db.connect() as conn:
        row, err = _customer_or_error(conn, customer_id)
        if err:
            return err
        open_tickets = conn.execute(
            "select count(*) from tickets where customer_id = ? and status != 'closed'", (customer_id,)
        ).fetchone()[0]
        out = {
            "name": row["name"],
            "account_type": row["account_type"],
            "base_currency": row["base_ccy"],
            "status": row["status"],
            "kyc_tier": row["kyc_tier"],
            "open_tickets": open_tickets,
            "has_open_positions": bool(row["open_positions"]),
        }
        if row["status"] == "restricted":
            # Deliberately vague: support must not disclose why an account is under review.
            out["note"] = "Your account is restricted pending a review. Support cannot share the reason; a compliance officer will contact you."
        return out


@mcp.tool()
def get_kyc_status(customer_id: str) -> dict:
    """Verification tier, status, and any documents still missing."""
    with db.connect() as conn:
        row, err = _customer_or_error(conn, customer_id)
        if err:
            return err
        next_tier = {1: "Tier 2 needs a proof of address", 2: "Tier 3 needs source-of-funds evidence", 3: "Highest tier reached"}
        return {
            "kyc_tier": row["kyc_tier"],
            "kyc_status": row["kyc_status"],
            "missing_documents": json.loads(row["missing_docs"]),
            "next_tier": next_tier[row["kyc_tier"]],
        }


@mcp.tool()
def get_withdrawal_limits(customer_id: str) -> dict:
    """Daily withdrawal limit and how many withdrawals this month (the first each month is free)."""
    with db.connect() as conn:
        row, err = _customer_or_error(conn, customer_id)
        if err:
            return err
        return {
            "daily_withdrawal_limit_usd": row["daily_limit"],
            "withdrawals_this_month": row["withdrawals_this_month"],
            "next_bank_wire_withdrawal_fee_usd": 0 if row["withdrawals_this_month"] == 0 else 10,
        }


@mcp.tool()
def list_tickets(customer_id: str) -> dict:
    """The customer's support tickets, newest first."""
    with db.connect() as conn:
        _, err = _customer_or_error(conn, customer_id)
        if err:
            return err
        rows = conn.execute(
            "select id, category, summary, status, created_at from tickets where customer_id = ? order by id desc limit 10",
            (customer_id,),
        ).fetchall()
        return {"tickets": [dict(r) for r in rows]}


@mcp.tool()
def create_ticket(customer_id: str, category: str, summary: str, priority: str = "normal") -> dict:
    """Log a support ticket. category: complaint | general | technical. Complaints are always logged."""
    if category not in {"complaint", "general", "technical"}:
        return {"error": "invalid_category"}
    with db.connect() as conn:
        _, err = _customer_or_error(conn, customer_id)
        if err:
            return err
        cur = conn.execute(
            "insert into tickets (customer_id, category, summary, priority, status, created_at) values (?,?,?,?,?,?)",
            (customer_id, category, summary[:500], priority if priority in {"normal", "high"} else "normal", "open", db.now()),
        )
        conn.commit()
        return {"ticket_id": cur.lastrowid, "status": "open", "category": category}


@mcp.tool()
def submit_limit_exception_request(customer_id: str, requested_daily_limit: int, reason: str, approved_by: str = "") -> dict:
    """SENSITIVE. Grant a temporary higher daily withdrawal limit. Refuses without a human reviewer's `approved_by`."""
    if not approved_by.strip():
        return {"error": "approval_required", "detail": "A compliance reviewer must approve this request first."}
    with db.connect() as conn:
        row, err = _customer_or_error(conn, customer_id)
        if err:
            return err
        tier = row["kyc_tier"]
        ceiling = db.TIER_LIMITS.get(tier + 1)
        if ceiling is None:
            return {"error": "already_highest_tier"}
        if requested_daily_limit <= row["daily_limit"] or requested_daily_limit > ceiling:
            return {"error": "outside_policy", "detail": f"An exception must be above {row['daily_limit']} and cannot exceed {ceiling} USD (the next tier's limit)."}
        conn.execute(
            "insert into limit_requests (customer_id, requested_limit, reason, approved_by, created_at) values (?,?,?,?,?)",
            (customer_id, requested_daily_limit, reason[:500], approved_by, db.now()),
        )
        conn.execute("update customers set daily_limit = ? where id = ?", (requested_daily_limit, customer_id))
        conn.commit()
        return {"status": "granted", "new_daily_limit_usd": requested_daily_limit, "approved_by": approved_by}


@mcp.tool()
def submit_account_closure_request(customer_id: str, reason: str, approved_by: str = "") -> dict:
    """SENSITIVE. Start account closure. Refuses without a human reviewer's `approved_by`."""
    if not approved_by.strip():
        return {"error": "approval_required", "detail": "A compliance reviewer must approve this request first."}
    with db.connect() as conn:
        row, err = _customer_or_error(conn, customer_id)
        if err:
            return err
        if row["open_positions"]:
            return {"error": "open_positions", "detail": "Open positions must be closed before an account can be closed."}
        conn.execute(
            "insert into closure_requests (customer_id, reason, approved_by, created_at) values (?,?,?,?)",
            (customer_id, reason[:500], approved_by, db.now()),
        )
        conn.execute("update customers set status = 'closure_pending' where id = ?", (customer_id,))
        conn.commit()
        return {"status": "closure_pending", "approved_by": approved_by}


if __name__ == "__main__":
    mcp.run(transport="stdio")
