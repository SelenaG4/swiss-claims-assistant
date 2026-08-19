"""Generates a synthetic Swiss claims dataset and trains the risk-scoring
model on it. There is no real insurer data behind this -- the point is to
demonstrate the feature engineering, training, and threshold-routing
architecture end to end, not to ship a validated fraud model. Labels are
assigned by a hand-specified rule (not by a real fraud investigation
outcome), stated explicitly so this can't be mistaken for real model
validation.

Usage:
    python scripts/generate_synthetic_claims.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# A bug hit while deploying, fixed here rather than papered over: in small/
# cgroup-limited containers (Render's free tier included), joblib's loky
# backend can miscount physical CPU cores as 0 and raise ValueError before
# scikit-learn even starts training. Setting this explicitly bypasses that
# broken autodetection. Must be set before numpy/pandas/sklearn are imported.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "2")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.risk import save_model, train  # noqa: E402

N_SAMPLES = 4000
RNG_SEED = 42


def generate_dataset(n: int = N_SAMPLES, seed: int = RNG_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    claim_amount_chf = rng.lognormal(mean=6.5, sigma=1.1, size=n).round(2)  # skewed, most claims small-ish
    days_since_policy_start = rng.integers(1, 3650, size=n)  # up to ~10 years
    prior_claims_count = rng.poisson(0.6, size=n)
    documentation_completeness_score = rng.beta(a=5, b=2, size=n)  # skewed toward "mostly complete"
    canton_risk_index = rng.uniform(0, 1, size=n)  # synthetic proxy, not a real cantonal statistic

    # Labeling rule (synthetic ground truth, not real fraud outcomes): risk
    # rises with claim size, very new policies, and prior-claims history;
    # falls with documentation completeness. Noise added so it isn't
    # trivially separable.
    risk_signal = (
        0.35 * (claim_amount_chf / claim_amount_chf.max())
        + 0.25 * (days_since_policy_start < 30).astype(float)  # brand-new policy = higher risk
        + 0.20 * np.minimum(prior_claims_count / 3, 1.0)
        + 0.20 * (1 - documentation_completeness_score)
        + 0.10 * canton_risk_index
        + rng.normal(0, 0.08, size=n)
    )
    is_high_risk = (risk_signal > np.quantile(risk_signal, 0.8)).astype(int)  # top ~20% flagged

    return pd.DataFrame(
        {
            "claim_amount_chf": claim_amount_chf,
            "days_since_policy_start": days_since_policy_start,
            "prior_claims_count": prior_claims_count,
            "documentation_completeness_score": documentation_completeness_score,
            "canton_risk_index": canton_risk_index,
            "is_high_risk": is_high_risk,
        }
    )


def main() -> None:
    df = generate_dataset()
    out_path = ROOT / "data" / "synthetic_claims.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} synthetic claims to {out_path} ({df['is_high_risk'].mean():.1%} flagged high-risk)")

    model, auc = train(df)
    save_model(model)
    print(f"Trained HistGradientBoostingClassifier, held-out AUC = {auc:.3f}, saved to app/../models/risk_model.joblib")


if __name__ == "__main__":
    main()
