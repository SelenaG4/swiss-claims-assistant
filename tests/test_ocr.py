from pathlib import Path

import pytest

from app.ocr import extract_invoice_fields
from scripts.generate_sample_invoice import generate

FIXTURE_DIR = Path(__file__).resolve().parent / "_fixtures"


@pytest.fixture(scope="module")
def invoice_image():
    path = FIXTURE_DIR / "test_invoice.png"
    generate(path, amount="1'240.50", date="03.02.2026", ref="INV-2026-00871")
    return path


def test_extracts_amount_date_and_reference(invoice_image):
    result = extract_invoice_fields(str(invoice_image))
    assert result.amount_chf == pytest.approx(1240.50)
    assert result.date == "03.02.2026"
    assert result.reference_number == "INV-2026-00871"


def test_raw_text_is_populated(invoice_image):
    result = extract_invoice_fields(str(invoice_image))
    assert "Gesamtbetrag" in result.raw_text
