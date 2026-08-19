from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app
from scripts.generate_sample_invoice import generate

FIXTURE_DIR = Path(__file__).resolve().parent / "_fixtures"


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def invoice_path():
    path = FIXTURE_DIR / "api_test_invoice.png"
    generate(path, amount="450.00", date="10.01.2026", ref="INV-2026-01001")
    return path


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_ingest_claim_redacts_pii_and_returns_routing(client):
    description = "Herr Peter Muster meldet einen Wasserschaden. AHV-Nr: 756.1234.5678.97"
    resp = client.post(
        "/claims/ingest",
        data={
            "description": description,
            "language": "de",
            "prior_claims_count": 0,
            "days_since_policy_start": 900,
            "canton_risk_index": 0.1,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "756.1234.5678.97" not in body["redacted_description"]
    assert body["pii_entities_redacted"]["AHV_NUMBER"] == 1
    assert body["routing"] in ("auto_approve", "flag_for_adjuster")


def test_ingest_claim_with_invoice_extracts_amount(client, invoice_path):
    with open(invoice_path, "rb") as f:
        resp = client.post(
            "/claims/ingest",
            data={"description": "Small claim, clean documentation.", "language": "en"},
            files={"invoice_image": ("invoice.png", f, "image/png")},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["invoice"]["amount_chf"] == pytest.approx(450.0)
    assert body["invoice"]["reference_number"] == "INV-2026-01001"


def test_get_claim_roundtrip(client):
    resp = client.post("/claims/ingest", data={"description": "Routine claim, no issues."})
    claim_id = resp.json()["id"]
    got = client.get(f"/claims/{claim_id}")
    assert got.status_code == 200
    assert got.json()["id"] == claim_id


def test_get_unknown_claim_returns_404(client):
    assert client.get("/claims/999999").status_code == 404


def test_policy_query_cross_lingual(client):
    resp = client.post("/policy-query", json={"query": "What is the deductible for water damage?"})
    assert resp.status_code == 200
    results = resp.json()["results"]
    assert results
    assert results[0]["clause_id"] == "DE-DEDUCTIBLE-01"


def test_policy_query_translates_answer_into_requested_language(client):
    resp = client.post(
        "/policy-query",
        json={"query": "What is the deductible for water damage?", "answer_language": "fr"},
    )
    assert resp.status_code == 200
    top = resp.json()["results"][0]
    assert top["source_language"] == "de"
    assert top["translation_mode"] == "reference_translation"
    assert "translated_text" in top
    assert top["translated_text"] != top["text"]
    assert "franchise" in top["translated_text"].lower()


def test_policy_query_skips_translation_when_already_in_requested_language(client):
    resp = client.post(
        "/policy-query",
        json={"query": "What is the deductible for water damage?", "answer_language": "de"},
    )
    assert resp.status_code == 200
    top = resp.json()["results"][0]
    assert top["translation_mode"] == "already_in_requested_language"
    assert top["translated_text"] == top["text"]


def test_summary_reflects_ingested_claims(client):
    resp = client.get("/reports/summary")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total_claims"] >= 3
    assert body["auto_approved"] + body["flagged_for_adjuster"] == body["total_claims"]
