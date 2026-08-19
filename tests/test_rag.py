import json
from pathlib import Path

import pytest

from app.rag import MultilingualPolicyIndex, PolicyClause

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "policy_clauses.json"


@pytest.fixture()
def index():
    clauses = [PolicyClause(**c) for c in json.load(open(DATA_PATH, encoding="utf-8"))]
    return MultilingualPolicyIndex(clauses)


def test_french_query_finds_german_deductible_clause(index):
    results = index.search("Quelle est la franchise en cas de dégât d'eau ?")
    assert results, "expected at least one match"
    assert results[0]["clause_id"] == "DE-DEDUCTIBLE-01"
    assert results[0]["source_language"] == "de"


def test_english_query_finds_german_deductible_clause(index):
    results = index.search("What is the deductible for water damage?")
    assert results[0]["clause_id"] == "DE-DEDUCTIBLE-01"


def test_german_query_finds_english_suva_clause(index):
    results = index.search("Wie funktioniert die Unfallversicherung SUVA?")
    assert results[0]["clause_id"] == "EN-SUVA-01"


def test_irrelevant_query_returns_no_high_confidence_match(index):
    results = index.search("xyloponic gravitational widget", top_k=3)
    assert results == [] or all(r["score"] < 0.05 for r in results)


def test_empty_index_returns_no_results():
    empty = MultilingualPolicyIndex([])
    assert empty.search("franchise") == []


def test_search_results_include_reference_translations(index):
    results = index.search("What is the deductible for water damage?")
    top = results[0]
    assert top["clause_id"] == "DE-DEDUCTIBLE-01"
    assert "fr" in top["translations"]
    assert "en" in top["translations"]
    assert "franchise" in top["translations"]["fr"].lower()
