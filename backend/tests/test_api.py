"""End-to-end API behaviour, including the full ingest -> ask round trip."""

import io
import json

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(app_db):
    from app.main import app

    return TestClient(app)


DOC = b"""# Orion Freight Logistics Platform

## Executive Summary
Orion Freight asked us to replace their dispatch system. The rollout covered 900 vehicles.

## Technology Stack
Built in Go with Kafka and PostgreSQL on Google Cloud.

## Commercials
This is a fixed bid of USD 275,000 delivered over 6 months in 2024.
"""


def _upload(client, name="orion_proposal.md", body=DOC):
    return client.post("/api/documents", files={"file": (name, io.BytesIO(body), "text/markdown")})


def test_health_reports_the_active_backends(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["llm"] and body["embeddings"]


def test_upload_then_ask_round_trip(client):
    response = _upload(client)
    assert response.status_code == 200
    doc_id = response.json()["document_id"]

    # Ingestion runs on a background thread; wait for it to settle.
    for _ in range(100):
        document = client.get(f"/api/documents/{doc_id}").json()
        if document["status"] in {"ready", "failed"}:
            break
    assert document["status"] == "ready", document.get("error")
    assert document["chunk_count"] > 0
    assert document["value_usd"] == 275_000
    assert document["engagement_model"] == "fixed_bid"
    assert "go" in document["tech_stack"] and "kafka" in document["tech_stack"]

    answer = client.post("/api/ask", json={"question": "What was the fixed bid for Orion Freight?"}).json()
    assert answer["conversation_id"] > 0
    assert answer["citations"], "an answer must cite its evidence"
    assert answer["invalid_citations"] == []
    assert answer["retrieval"]["intent"] == "price"


def test_uploading_the_same_file_twice_is_rejected(client):
    assert _upload(client).status_code == 200
    duplicate = _upload(client, name="different_name.md")
    assert duplicate.status_code == 409
    assert "already in the library" in duplicate.json()["detail"]


def test_unsupported_file_type_is_rejected_before_ingestion(client):
    response = client.post(
        "/api/documents", files={"file": ("notes.xlsx", io.BytesIO(b"binary"), "application/vnd.ms-excel")}
    )
    assert response.status_code == 415
    assert client.get("/api/documents").json() == []


def test_empty_file_is_rejected(client):
    response = client.post("/api/documents", files={"file": ("empty.md", io.BytesIO(b""), "text/markdown")})
    assert response.status_code == 400


def test_asking_with_an_empty_library_does_not_fabricate(client):
    answer = client.post("/api/ask", json={"question": "What did we quote for anything?"}).json()
    assert answer["citations"] == []
    assert "nothing in the indexed library" in answer["answer"].lower()


def test_conversation_history_is_persisted(client, ):
    _upload(client)
    for _ in range(100):
        if client.get("/api/documents").json()[0]["status"] in {"ready", "failed"}:
            break

    first = client.post("/api/ask", json={"question": "What did Orion cost?"}).json()
    conversation_id = first["conversation_id"]
    client.post("/api/ask", json={"question": "and how long did it take?", "conversation_id": conversation_id})

    messages = client.get(f"/api/conversations/{conversation_id}/messages").json()
    assert [m["role"] for m in messages] == ["user", "assistant", "user", "assistant"]


def test_stream_emits_meta_then_tokens_then_done(client):
    _upload(client)
    for _ in range(100):
        if client.get("/api/documents").json()[0]["status"] in {"ready", "failed"}:
            break

    events: list[str] = []
    payloads: dict[str, dict] = {}
    with client.stream("POST", "/api/ask/stream", json={"question": "What was the Orion price?", "mode": "answer"}) as response:
        assert response.status_code == 200
        current = None
        for line in response.iter_lines():
            if line.startswith("event:"):
                current = line.split(":", 1)[1].strip()
                events.append(current)
            elif line.startswith("data:") and current in {"meta", "done"}:
                payloads[current] = json.loads(line.split(":", 1)[1].strip())

    assert events[0] == "meta"
    assert events[-1] == "done"
    assert "token" in events
    # Citations arrive with `meta`, before the answer streams, so the sources
    # panel can populate while the answer is still being written.
    assert "citations" in payloads["meta"]


def test_deleting_a_document_removes_its_vectors(client):
    doc_id = _upload(client).json()["document_id"]
    for _ in range(100):
        if client.get(f"/api/documents/{doc_id}").json()["status"] in {"ready", "failed"}:
            break

    before = client.get("/api/stats").json()
    assert before["vectors"] > 0

    assert client.delete(f"/api/documents/{doc_id}").status_code == 200
    after = client.get("/api/stats").json()
    assert after["vectors"] == 0 and after["chunks"] == 0


def test_facets_describe_the_indexed_corpus(client):
    _upload(client)
    for _ in range(100):
        if client.get("/api/documents").json()[0]["status"] in {"ready", "failed"}:
            break
    facets = client.get("/api/facets").json()
    assert {"value": "logistics", "count": 1} in facets["industries"]
    assert any(t["value"] == "kafka" for t in facets["tech"])


def test_eval_harness_runs_and_reports_metrics(client):
    _upload(client)
    for _ in range(100):
        if client.get("/api/documents").json()[0]["status"] in {"ready", "failed"}:
            break
    client.post("/api/eval/seed")
    result = client.post("/api/eval/run", json={"k": 6, "generate_answers": True}).json()
    metrics = result["metrics"]
    assert metrics["cases"] > 0
    assert "mrr" in metrics and "recall_at_6" in metrics
    assert client.get("/api/eval/runs").json()[0]["metrics"]["cases"] == metrics["cases"]
