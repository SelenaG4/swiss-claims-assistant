import pytest

from app.risk import assess, train
from scripts.generate_synthetic_claims import generate_dataset


@pytest.fixture(scope="module")
def model():
    df = generate_dataset(n=2000, seed=7)
    trained_model, auc = train(df)
    assert auc > 0.6  # sanity check: better than a coin flip on held-out data
    return trained_model


def test_small_clean_claim_is_auto_approved(model):
    result = assess(
        model,
        claim_amount_chf=250.0,
        days_since_policy_start=900,  # long-standing policy
        prior_claims_count=0,
        documentation_completeness_score=0.98,
        canton_risk_index=0.1,
    )
    assert result.risk_score <= 20
    assert result.routing == "auto_approve"


def test_large_new_policy_claim_is_flagged(model):
    result = assess(
        model,
        claim_amount_chf=45000.0,
        days_since_policy_start=5,  # brand-new policy
        prior_claims_count=3,
        documentation_completeness_score=0.2,
        canton_risk_index=0.9,
    )
    assert result.routing == "flag_for_adjuster"
    assert result.risk_score > 20


def test_large_claim_is_flagged_even_with_low_score(model):
    # Amount alone should force adjuster review regardless of the model score.
    result = assess(
        model,
        claim_amount_chf=50000.0,
        days_since_policy_start=1500,
        prior_claims_count=0,
        documentation_completeness_score=0.99,
        canton_risk_index=0.05,
    )
    assert result.routing == "flag_for_adjuster"


def test_risk_score_is_bounded(model):
    result = assess(model, 500.0, 100, 1, 0.7, 0.5)
    assert 0 <= result.risk_score <= 100
