"""Renders a synthetic invoice image as a test fixture -- no real Swiss
insurer documents were available for this project, so OCR is exercised
against a generated image with a realistic layout instead. This is stated
explicitly rather than hidden, same as the other honesty notes in this repo.

Usage:
    python scripts/generate_sample_invoice.py
"""
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def generate(path: Path, amount: str = "850.00", date: str = "14.08.2026", ref: str = "POL-2026-00193") -> None:
    img = Image.new("RGB", (600, 400), color="white")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
        font_bold = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
    except OSError:
        font = ImageFont.load_default()
        font_bold = font

    draw.text((30, 20), "Schadensrechnung / Facture de sinistre", fill="black", font=font_bold)
    draw.text((30, 80), f"Datum: {date}", fill="black", font=font)
    draw.text((30, 120), f"Referenz: {ref}", fill="black", font=font)
    draw.text((30, 160), "Position: Wasserschaden Reparatur Kueche", fill="black", font=font)
    draw.text((30, 220), f"Gesamtbetrag: CHF {amount}", fill="black", font=font_bold)
    draw.text((30, 280), "Zahlbar innert 30 Tagen.", fill="black", font=font)

    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    print(f"Wrote {path}")


if __name__ == "__main__":
    generate(ROOT / "data" / "sample_invoice.png")
