import pytest

from app.tools import McpToolGateway, ToolError
from tests.conftest import run


def test_model_only_sees_read_tools_without_identity_args(mock_db):
    async def go():
        async with McpToolGateway(mock_db) as gw:
            names = {s["function"]["name"] for s in gw.model_tool_specs}
            assert names == {"get_account_summary", "get_kyc_status", "get_withdrawal_limits", "list_tickets"}
            for spec in gw.model_tool_specs:
                assert "customer_id" not in spec["function"]["parameters"].get("properties", {})
    run(go())


def test_customer_id_cannot_be_overridden_by_the_model(mock_db):
    async def go():
        async with McpToolGateway(mock_db) as gw:
            res = await gw.call("get_account_summary", "C-1001", {"customer_id": "C-1002"})
            assert res["name"] == "Aisha Rahman"
    run(go())


def test_sensitive_tools_are_not_callable_on_the_normal_path(mock_db):
    async def go():
        async with McpToolGateway(mock_db) as gw:
            with pytest.raises(ToolError):
                await gw.call("submit_limit_exception_request", "C-1004", {"requested_daily_limit": 5000, "reason": "x"})
    run(go())


def test_server_refuses_sensitive_writes_without_approval(mock_db):
    async def go():
        async with McpToolGateway(mock_db) as gw:
            res = await gw.call("submit_limit_exception_request", "C-1004", {"requested_daily_limit": 5000, "reason": "x"}, privileged=True)
            assert res["error"] == "approval_required"
            ok = await gw.call("submit_limit_exception_request", "C-1004", {"requested_daily_limit": 5000, "reason": "x"}, approved_by="r1", privileged=True)
            assert ok["status"] == "granted"
    run(go())


def test_policy_ceiling_is_enforced_server_side(mock_db):
    async def go():
        async with McpToolGateway(mock_db) as gw:
            res = await gw.call("submit_limit_exception_request", "C-1004", {"requested_daily_limit": 50000, "reason": "x"}, approved_by="r1", privileged=True)
            assert res["error"] == "outside_policy"
    run(go())


def test_restricted_account_reason_is_not_disclosed(mock_db):
    async def go():
        async with McpToolGateway(mock_db) as gw:
            res = await gw.call("get_account_summary", "C-1003")
            assert res["status"] == "restricted" and "cannot share the reason" in res["note"]
    run(go())


def test_unknown_customer(mock_db):
    async def go():
        async with McpToolGateway(mock_db) as gw:
            assert (await gw.call("get_kyc_status", "C-9999"))["error"] == "customer_not_found"
    run(go())
