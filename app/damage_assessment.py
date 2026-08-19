"""Visual damage assessment from a claim photo -- deliberately NOT a trained
model.

Building a real damage-severity classifier needs a labeled dataset of actual
claims photos tied to actual assessed repair costs. No such dataset exists
for this project (and none should be assembled from real policyholder
photos without their consent and a data-processing agreement, which is
exactly the kind of governance question this whole project is about). Rather
than fake a result or quietly skip this stage, it's implemented as an
explicit, clearly-labeled placeholder that documents the real interface this
would need to satisfy in production -- the same honesty pattern used for the
untrained CNN weights in the Emotion Insight Assistant project: surface the
gap in the API response (`mode`), never hide it behind a plausible-looking
number.
"""
from __future__ import annotations

from dataclasses import dataclass

from PIL import Image, ImageStat


@dataclass
class DamageAssessment:
    severity_estimate: str  # "unknown" -- see docstring
    confidence: float
    mode: str
    note: str


def assess_damage_photo(image_path: str) -> DamageAssessment:
    """Placeholder implementation. Does read and validate the image (so the
    pipeline can at least confirm a usable photo was submitted), but does NOT
    estimate real damage severity -- that requires a trained model this
    project doesn't have the data to build honestly."""
    image = Image.open(image_path)
    image.verify()  # confirms it's a real, non-corrupt image file

    return DamageAssessment(
        severity_estimate="unknown",
        confidence=0.0,
        mode="placeholder_not_trained",
        note=(
            "Damage-severity scoring is not implemented -- no labeled Swiss claims-photo "
            "dataset was available to train it honestly. In production this stage would be "
            "a CV model trained on real, consented claims imagery, scoring severity the same "
            "way app/risk.py scores fraud risk: as one more feature feeding the routing "
            "decision, not a replacement for it."
        ),
    )
