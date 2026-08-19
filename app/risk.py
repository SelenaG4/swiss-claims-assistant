"""Automated risk/fraud triaging: scores a claim 0-100 and routes it, so
low-risk, low-value claims get fast-tracked and the expensive human review
time (CHF 120-220+/hour in Switzerland) is spent only on claims that actually
need it.

Uses scikit-learn's HistGradientBoostingClassifier -- the same family of
model as XGBoost (gradient-boosted decision trees), chosen here instead of
XGBoost specifically to keep this portfolio project's dependency footprint
small (no CUDA/GPU packages pulled in for a CPU-only demo). Swapping in
XGBoost or LightGBM would be a one-line change; the feature engineering and
routing logic are what actually matter here, not the specific library.

Trained on a synthetic dataset (scripts/generate_synthetic_claims.py) with
deliberately realistic feature distributions and a hand-specified risk rule
used to *label* the synthetic data -- there is no real Swiss insurer claims
dataset behind this. That's stated plainly here and in the README: this
demonstrates the modeling and routing architecture, not a validated
production fraud model, which would need real labeled claims history.
"""
from __future__ import annotations

import os

# A bug hit while deploying (Render's free tier container), fixed here rather
# than in just one entry point: joblib's loky backend can miscount physical
# CPU cores as 0 in small/cgroup-limited containers and raise ValueError
# before scikit-learn ever gets to train. Set before joblib/sklearn are
# imported below, so it's fixed regardless of whether this module is first
# reached via the training script or via the API server's startup fallback.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "2")

from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

FEATURE_COLUMNS = [
    "claim_amount_chf",
    "days_since_policy_start",
    "prior_claims_count",
    "documentation_completeness_score",
    "canton_risk_index",
]

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "risk_model.joblib"

# Below this score AND this claim amount, a claim is auto-approved without
# a human adjuster. Both conditions matter -- a small claim with a very
# suspicious feature profile still gets flagged.
AUTO_APPROVE_MAX_SCORE = 20
AUTO_APPROVE_MAX_AMOUNT_CHF = 1000.0


@dataclass
class RiskAssessment:
    risk_score: int  # 0-100, higher = riskier
    routing: str  # "auto_approve" | "flag_for_adjuster"
    reason: str


def train(df: pd.DataFrame) -> tuple[HistGradientBoostingClassifier, float]:
    X = df[FEATURE_COLUMNS]
    y = df["is_high_risk"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = HistGradientBoostingClassifier(random_state=42)
    model.fit(X_train, y_train)

    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    return model, auc


def save_model(model: HistGradientBoostingClassifier) -> None:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)


def load_model() -> HistGradientBoostingClassifier | None:
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


def assess(
    model: HistGradientBoostingClassifier,
    claim_amount_chf: float,
    days_since_policy_start: int,
    prior_claims_count: int,
    documentation_completeness_score: float,
    canton_risk_index: float,
) -> RiskAssessment:
    row = pd.DataFrame(
        [[claim_amount_chf, days_since_policy_start, prior_claims_count, documentation_completeness_score, canton_risk_index]],
        columns=FEATURE_COLUMNS,
    )
    probability = float(model.predict_proba(row)[0, 1])
    score = round(probability * 100)

    if score <= AUTO_APPROVE_MAX_SCORE and claim_amount_chf < AUTO_APPROVE_MAX_AMOUNT_CHF:
        return RiskAssessment(
            risk_score=score,
            routing="auto_approve",
            reason=f"Score {score} <= {AUTO_APPROVE_MAX_SCORE} and amount CHF {claim_amount_chf:.2f} < {AUTO_APPROVE_MAX_AMOUNT_CHF:.0f}",
        )
    return RiskAssessment(
        risk_score=score,
        routing="flag_for_adjuster",
        reason=f"Score {score} or amount CHF {claim_amount_chf:.2f} exceeds fast-track threshold",
    )
