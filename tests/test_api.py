import pytest
from fastapi.testclient import TestClient

from app.api import create_app
from app.llm import OfflineLLM
from tests.conftest import run


@pytest.fixture
def client(retriever, mock_db):
    with TestClient(create_app(llm=OfflineLLM(), retriever=retriever, mock_db_path=mock_db)) as c:
        yield c


def chat(client, message, session="s1", customer="C-1004"):
    return client.post("/chat", json={"session_id": session, "customer_id": customer, "message": message})


def test_health_reports_the_setup(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["llm"] == "offline-rules"
    assert "submit_limit_exception_request" not in body["model_visible_tools"]


def test_customers_list_has_no_account_data(client):
    rows = client.get("/customers").json()
    assert {r["id"] for r in rows} >= {"C-1001", "C-1004"} and all(set(r) == {"id", "name", "account_type"} for r in rows)


def test_chat_faq(client):
    r = chat(client, "What is the withdrawal fee?").json()
    assert r["status"] == "done" and r["sources"] and r["trace"]


def test_full_human_approval_flow(client):
    chat(client, "Can you increase my daily withdrawal limit?")
    chat(client, "5000")
    r = chat(client, "I need it for my university tuition fee next week").json()
    assert r["status"] == "pending_approval"
    # the session can't be used for something else while it waits
    assert chat(client, "hello").json()["status"] == "pending_approval"
    assert client.get("/sessions/s1").json()["status"] == "pending_approval"

    pending = client.get("/approvals").json()
    assert len(pending) == 1 and pending[0]["action"]["requested_daily_limit"] == 5000

    done = client.post(f"/approvals/{pending[0]['approval_id']}/decision", json={"approved": True, "reviewer": "reviewer-7"}).json()
    assert "now 5000 USD" in done["reply"]
    assert client.get("/approvals").json() == []
    assert client.get("/sessions/s1").json()["status"] == "done"
    assert client.post("/approvals/s1/decision", json={"approved": True, "reviewer": "x"}).status_code == 404


def test_decisions_carry_trace_ids_and_are_listed_for_audit(client):
    chat(client, "Can you increase my daily withdrawal limit?")
    chat(client, "5000")
    chat(client, "I need it for my university tuition fee next week")
    pending = client.get("/approvals").json()[0]
    assert pending["trace_id"] and pending["requested_at"]

    done = client.post(f"/approvals/{pending['approval_id']}/decision", json={"approved": False, "reviewer": "r9", "note": "not now"})
    rec = done.json()["audit"]
    assert rec["decision"] == "declined" and rec["reviewer"] == "r9" and rec["action"] == "limit_exception"
    assert rec["request_trace_id"] == pending["trace_id"]
    assert rec["decision_trace_id"] == done.headers["X-Request-ID"]
    assert rec["requested_at"] <= rec["decided_at"]
    assert client.get("/approvals/audit").json()[0] == rec


def test_trace_steps_are_timed(client):
    trace = chat(client, "What is the withdrawal fee?").json()["trace"]
    timed = [s for s in trace if s["step"] != "done"]
    assert timed and all(isinstance(s.get("ms"), int) for s in timed)
    hits = next(s for s in trace if s["step"] == "retrieve")["hits"]
    assert hits and all(h["snippet"] for h in hits)


def test_bad_input_is_rejected_by_validation(client):
    assert client.post("/chat", json={"session_id": "bad id!", "message": "hi"}).status_code == 422
    assert client.post("/chat", json={"session_id": "s", "customer_id": "../etc", "message": "hi"}).status_code == 422


def test_rate_limit(retriever, mock_db, monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "3")
    from app.config import get_settings
    get_settings.cache_clear()
    try:
        with TestClient(create_app(llm=OfflineLLM(), retriever=retriever, mock_db_path=mock_db)) as c:
            codes = [chat(c, "What is the withdrawal fee?", session=f"r{i}").status_code for i in range(5)]
        assert codes[:3] == [200, 200, 200] and codes[3] == 429
    finally:
        get_settings.cache_clear()


def test_api_key_is_enforced_when_configured(retriever, mock_db, monkeypatch):
    monkeypatch.setenv("API_KEY", "secret")
    monkeypatch.setenv("REVIEWER_KEY", "rev")
    from app.config import get_settings
    get_settings.cache_clear()
    try:
        with TestClient(create_app(llm=OfflineLLM(), retriever=retriever, mock_db_path=mock_db)) as c:
            body = {"session_id": "k1", "customer_id": "C-1001", "message": "What is the withdrawal fee?"}
            assert c.post("/chat", json=body).status_code == 401
            assert c.post("/chat", json=body, headers={"X-API-Key": "secret"}).status_code == 200
            assert c.get("/approvals").status_code == 401
            assert c.get("/approvals", headers={"X-Reviewer-Key": "rev"}).status_code == 200
    finally:
        get_settings.cache_clear()


def test_index_page_is_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "Aurelia" in r.text and "text/html" in r.headers["content-type"]


def test_daily_cap_stops_chat(client):
    client.app.state.daily_cap.limit = 2
    assert chat(client, "What is the withdrawal fee?", session="cap1").status_code == 200
    assert chat(client, "What is the withdrawal fee?", session="cap2").status_code == 200
    r = chat(client, "What is the withdrawal fee?", session="cap3")
    assert r.status_code == 429 and "limit" in r.json()["detail"]


def test_reset_restores_seed_data(client):
    chat(client, "Raise my daily withdrawal limit to 5000 USD because of university tuition", session="rs")
    client.post("/approvals/rs/decision", json={"approved": True, "reviewer": "r1"})
    after = chat(client, "What is my withdrawal limit?", session="rs2").json()["reply"]
    assert "5000" in after
    assert client.post("/demo/reset").json() == {"status": "reset"}
    assert client.get("/approvals").json() == []
    again = chat(client, "What is my withdrawal limit?", session="rs3").json()["reply"]
    assert "2000" in again and "5000" not in again


def test_session_returns_trace_after_decision(client):
    chat(client, "Raise my daily withdrawal limit to 5000 USD because of university tuition", session="tr")
    client.post("/approvals/tr/decision", json={"approved": False, "reviewer": "r1", "note": "not now"})
    s = client.get("/sessions/tr").json()
    assert s["status"] == "done" and any(step["step"] == "human_decision" for step in s["trace"])


def _audit(client):
    run(client.app.state.audit.drain())
    return list(client.app.state.audit.sink.records)


def test_chat_is_audited_with_request_and_response(client):
    r = chat(client, "What is the withdrawal fee?", session="aud1")
    rec = _audit(client)[-1]
    assert rec["path"] == "/chat" and rec["status_code"] == 200 and rec["session_id"] == "aud1"
    assert rec["request"]["message"] == "What is the withdrawal fee?"
    assert rec["response"]["reply"] == r.json()["reply"] and rec["route"] == "faq"
    assert rec["request_id"] == r.headers["x-request-id"] and rec["latency_ms"] >= 0


def test_polling_gets_are_not_audited_but_errors_are(client):
    before = len(_audit(client))
    client.get("/health"), client.get("/approvals"), client.get("/customers")
    assert len(_audit(client)) == before
    client.get("/sessions/does-not-exist")
    rec = _audit(client)[-1]
    assert rec["status_code"] == 404 and rec["method"] == "GET"


def test_audit_never_stores_headers_or_keys(client):
    client.post("/chat", json={"session_id": "aud3", "customer_id": "C-1004", "message": "hi"}, headers={"X-API-Key": "secret-xyz"})
    assert "secret-xyz" not in repr(_audit(client))


def test_decision_is_audited_against_the_session(client):
    chat(client, "Raise my daily withdrawal limit to 5000 USD because of university tuition", session="aud4")
    client.post("/approvals/aud4/decision", json={"approved": True, "reviewer": "r1"})
    rec = _audit(client)[-1]
    assert rec["path"] == "/approvals/aud4/decision" and rec["session_id"] == "aud4" and rec["request"]["reviewer"] == "r1"


def test_bad_database_url_degrades_instead_of_crashing(monkeypatch, mock_db):
    monkeypatch.setenv("VECTOR_STORE", "pgvector")
    monkeypatch.setenv("AUDIT_LOG", "postgres")
    monkeypatch.setenv("EMBEDDER", "lexical")
    monkeypatch.setenv("DATABASE_URL", "postgresql://nobody:wrong@127.0.0.1:1/none")
    from app.config import get_settings
    from app.rag.embedders import set_embedder
    get_settings.cache_clear()
    set_embedder(None)
    try:
        with TestClient(create_app(llm=OfflineLLM(), mock_db_path=mock_db)) as c:
            h = c.get("/health").json()
            assert h["vector_store"] == "memory" and h["request_log"] == "off" and len(h["warnings"]) == 2
            assert c.post("/chat", json={"session_id": "deg", "message": "What is the withdrawal fee?"}).json()["status"] == "done"
    finally:
        get_settings.cache_clear()
        set_embedder(None)
