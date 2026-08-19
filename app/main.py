"""Swiss Multilingual Claims & Compliance Assistant.

Pipeline: PII redaction -> OCR invoice extraction + (placeholder) damage
photo check -> multilingual policy RAG -> classical-ML risk triage ->
auto-approve or flag for a human adjuster. See README.md for what's real,
what's a documented placeholder, and why.

Run locally:
    pip install -r requirements.txt
    python scripts/generate_synthetic_claims.py   # trains + saves the risk model
    python scripts/generate_sample_invoice.py      # sample invoice image
    uvicorn app.main:app --reload

Then open http://localhost:8000/ -- a plain-language demo page (no API
knowledge needed): submit a sample claim and see personal data get redacted,
a risk score, and a routing decision; ask a policy question in DE/FR/EN and
see it matched across languages. Engineers/reviewers who want the raw API
instead can go straight to /docs (interactive Swagger UI, every form
pre-filled with a working example):
    POST /claims/ingest       -- submit a claim (text + optional invoice/photo)
    GET  /claims/{id}         -- the full pipeline result
    POST /policy-query        -- ask a policy question in DE/FR/EN
    GET  /reports/summary     -- routing breakdown across all submitted claims
"""
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app import pii, risk, translate
from app.damage_assessment import assess_damage_photo
from app.ocr import extract_invoice_fields
from app.rag import MultilingualPolicyIndex, PolicyClause

ROOT = Path(__file__).resolve().parent.parent

app = FastAPI(
    title="Swiss Multilingual Claims & Compliance Assistant",
    description=(
        "PII redaction (nFADP-aligned) -> OCR + damage-photo intake -> multilingual "
        "(DE/FR/EN) policy RAG -> classical-ML risk triage -> auto-approve or flag "
        "for a human adjuster."
    ),
    version="0.1.0",
)

_CLAIMS: dict[int, dict] = {}
_NEXT_ID = {"value": 1}
_policy_index: MultilingualPolicyIndex | None = None
_risk_model = None


@app.on_event("startup")
def _startup() -> None:
    global _policy_index, _risk_model
    clauses_path = ROOT / "data" / "policy_clauses.json"
    clauses = [PolicyClause(**c) for c in json.load(open(clauses_path, encoding="utf-8"))]
    _policy_index = MultilingualPolicyIndex(clauses)
    _risk_model = risk.load_model()
    if _risk_model is None:
        # Dev convenience: train on the fly if the model wasn't pre-generated.
        from scripts.generate_synthetic_claims import generate_dataset

        trained, _auc = risk.train(generate_dataset())
        risk.save_model(trained)
        _risk_model = trained


def _save_upload(upload: UploadFile) -> str:
    suffix = Path(upload.filename or "upload").suffix or ".png"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    shutil.copyfileobj(upload.file, tmp)
    tmp.close()
    return tmp.name


# Pre-filled so the interactive /docs page is a "click Execute" demo, not a
# blank form -- this example alone contains an AHV number, an IBAN, a
# person's name, and a date of birth, so hitting Execute immediately shows
# all four PII types getting redacted in the response.
_EXAMPLE_CLAIM_DESCRIPTION = (
    "Herr Matthias Keller meldet einen Wasserschaden in der Kueche, geboren am 12.03.1985. "
    "AHV-Nr: 756.1234.5678.97, IBAN: CH93 0076 2011 6238 5295 7. "
    "Kontakt: matthias.keller@example.ch. Schadensumme: CHF 850.00."
)


@app.get("/", include_in_schema=False)
def root() -> FileResponse:
    # The plain-language demo page -- for engineers/reviewers who want the
    # raw API instead, /docs (FastAPI's built-in Swagger UI) is always
    # available alongside it.
    return FileResponse(ROOT / "app" / "static" / "index.html")


@app.post("/claims/ingest")
def ingest_claim(
    description: str = Form(_EXAMPLE_CLAIM_DESCRIPTION),
    language: str = Form("de"),
    prior_claims_count: int = Form(0),
    days_since_policy_start: int = Form(900),
    canton_risk_index: float = Form(0.2),
    invoice_image: UploadFile | None = File(None),
    damage_photo: UploadFile | None = File(None),
) -> dict:
    redaction = pii.redact(description)

    invoice = None
    if invoice_image is not None:
        path = _save_upload(invoice_image)
        extraction = extract_invoice_fields(path)
        invoice = {
            "amount_chf": extraction.amount_chf,
            "date": extraction.date,
            "reference_number": extraction.reference_number,
        }

    damage = None
    if damage_photo is not None:
        path = _save_upload(damage_photo)
        assessment = assess_damage_photo(path)
        damage = {
            "severity_estimate": assessment.severity_estimate,
            "mode": assessment.mode,
            "note": assessment.note,
        }

    provided = [description, invoice_image, damage_photo]
    documentation_completeness_score = sum(1 for p in provided if p) / len(provided)

    claim_amount = invoice["amount_chf"] if invoice and invoice["amount_chf"] else 0.0

    assessment = risk.assess(
        _risk_model,
        claim_amount_chf=claim_amount,
        days_since_policy_start=days_since_policy_start,
        prior_claims_count=prior_claims_count,
        documentation_completeness_score=documentation_completeness_score,
        canton_risk_index=canton_risk_index,
    )

    claim_id = _NEXT_ID["value"]
    _NEXT_ID["value"] += 1
    record = {
        "id": claim_id,
        "language": language,
        "redacted_description": redaction.redacted_text,
        "pii_entities_redacted": redaction.entity_counts,
        "invoice": invoice,
        "damage_assessment": damage,
        "risk_score": assessment.risk_score,
        "routing": assessment.routing,
        "routing_reason": assessment.reason,
    }
    _CLAIMS[claim_id] = record
    return record


@app.get("/claims/{claim_id}")
def get_claim(claim_id: int) -> dict:
    record = _CLAIMS.get(claim_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Unknown claim_id {claim_id}")
    return record


class PolicyQueryIn(BaseModel):
    # Default is a cross-lingual demo on its own: a French query that only
    # matches a German-language clause (see app/rag.py) -- click Execute with
    # no changes and the top result is DE-DEDUCTIBLE-01.
    query: str = Field(default="Quelle est la franchise en cas de dégât d'eau ?")
    top_k: int = 3
    # If set, translate each result's clause text into this language before
    # returning it -- e.g. so a French-speaking adjuster who asked in French
    # gets the answer back in French even though the matching clause was only
    # ever written in German. `None` (the default) returns clauses in their
    # original source language, unchanged.
    answer_language: str | None = Field(default=None, examples=["fr"])


@app.post("/policy-query")
def policy_query(payload: PolicyQueryIn) -> dict:
    results = _policy_index.search(payload.query, top_k=payload.top_k)
    if payload.answer_language:
        for r in results:
            reference = r.get("translations", {}).get(payload.answer_language)
            if r["source_language"] == payload.answer_language:
                r["translated_text"] = r["text"]
                r["translation_mode"] = "already_in_requested_language"
            elif reference:
                # Hand-authored reference translation for this clause -- see
                # PolicyClause.translations docstring in app/rag.py. Fluent,
                # not machine-translated, and only possible because this is a
                # small closed set of demo clauses.
                r["translated_text"] = reference
                r["translation_mode"] = "reference_translation"
            else:
                # Fallback for any clause without a pre-authored translation
                # (shouldn't happen for this demo's fixed dataset, but keeps
                # the endpoint honest and functional if the dataset grows
                # without translations being added yet). Keyword-level
                # glossary substitution -- see app/translate.py docstring.
                translation = translate.translate(r["text"], payload.answer_language)
                r["translated_text"] = translation.text
                r["translation_mode"] = translation.mode
    return {"query": payload.query, "answer_language": payload.answer_language, "results": results}


@app.get("/reports/summary")
def summary() -> dict:
    total = len(_CLAIMS)
    auto_approved = sum(1 for c in _CLAIMS.values() if c["routing"] == "auto_approve")
    return {
        "total_claims": total,
        "auto_approved": auto_approved,
        "flagged_for_adjuster": total - auto_approved,
        "auto_approve_rate": round(auto_approved / total, 3) if total else None,
    }


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
