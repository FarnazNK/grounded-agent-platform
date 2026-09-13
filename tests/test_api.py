from __future__ import annotations

from io import BytesIO

from tests.conftest import auth_header


def test_protected_endpoint_requires_auth(client, seeded_app):
    response = client.get("/v1/documents", params={"workspace_id": seeded_app.alice_workspace_id})
    assert response.status_code == 401


def test_upload_query_and_list_documents(client, seeded_app):
    upload = client.post(
        "/v1/documents/upload",
        params={"workspace_id": seeded_app.alice_workspace_id},
        headers=auth_header(seeded_app.alice_id),
        files={
            "file": (
                "new_doc.md",
                BytesIO(b"Expense policy says receipts are required."),
                "text/markdown",
            )
        },
    )
    assert upload.status_code == 200
    body = upload.json()
    assert body["document"]["status"] == "ready"
    assert body["job"]["status"] == "completed"

    listing = client.get(
        "/v1/documents",
        params={"workspace_id": seeded_app.alice_workspace_id},
        headers=auth_header(seeded_app.alice_id),
    )
    assert listing.status_code == 200
    assert any(doc["source_name"] == "new_doc.md" for doc in listing.json()["documents"])

    query = client.post(
        "/v1/query",
        headers={**auth_header(seeded_app.alice_id), "x-request-id": "req-123"},
        json={
            "workspace_id": seeded_app.alice_workspace_id,
            "query": "What is the expense policy?",
        },
    )
    assert query.status_code == 200
    result = query.json()["result"]
    assert result["request_id"] == "req-123"
    assert result["citations"]
    assert result["chunks"]


def test_cross_tenant_access_is_forbidden(client, seeded_app):
    response = client.get(
        "/v1/documents",
        params={"workspace_id": seeded_app.alice_workspace_id},
        headers=auth_header(seeded_app.bob_id),
    )
    assert response.status_code == 403


def test_duplicate_ingestion_returns_conflict(client, seeded_app):
    response = client.post(
        "/v1/documents/upload",
        params={"workspace_id": seeded_app.alice_workspace_id},
        headers=auth_header(seeded_app.alice_id),
        files={
            "file": (
                "dup.md",
                BytesIO(b"Alpha PTO policy allows 20 vacation days and cites alpha only."),
                "text/markdown",
            )
        },
    )
    assert response.status_code == 409
    assert response.json()["error"] == "duplicate_ingestion"


def test_invalid_upload_is_rejected(client, seeded_app):
    response = client.post(
        "/v1/documents/upload",
        params={"workspace_id": seeded_app.alice_workspace_id},
        headers=auth_header(seeded_app.alice_id),
        files={"file": ("payload.exe", BytesIO(b"MZ"), "application/octet-stream")},
    )
    assert response.status_code == 400
    assert response.json()["error"] == "invalid_upload"
