"""Extracts structured fields (amount, date, invoice/policy number) from a
claim's invoice or receipt image via OCR -- the "bridge" a human currently
provides between an unstructured scanned document and a structured claims
record.

Uses pytesseract (real OCR, not mocked) against the actual pixels of
whatever image is passed in, then regex-parses Swiss-formatted amounts
(CHF 1'234.50), dates (DD.MM.YYYY), and policy/invoice reference numbers out
of the recognized text. Tested against synthetically rendered invoice images
(see tests/test_ocr.py and scripts/generate_sample_invoice.py) -- no real
Swiss insurer documents were available for this project, and none should be
used without the redaction layer (app/pii.py) running first regardless.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import pytesseract
from PIL import Image

# Swiss amount format: CHF 1'234.50 or CHF 850.00 (apostrophe as thousands separator).
_AMOUNT_RE = re.compile(r"CHF\s*([\d']+\.\d{2})")
_DATE_RE = re.compile(r"\b(\d{1,2}\.\d{1,2}\.\d{2,4})\b")
_REF_RE = re.compile(r"\b((?:POL|INV|REF)-[\w-]+)\b", re.IGNORECASE)


@dataclass
class InvoiceExtraction:
    raw_text: str
    amount_chf: float | None
    date: str | None
    reference_number: str | None


def extract_invoice_fields(image_path: str) -> InvoiceExtraction:
    image = Image.open(image_path)
    raw_text = pytesseract.image_to_string(image)

    amount = None
    m = _AMOUNT_RE.search(raw_text)
    if m:
        amount = float(m.group(1).replace("'", ""))

    date_match = _DATE_RE.search(raw_text)
    ref_match = _REF_RE.search(raw_text)

    return InvoiceExtraction(
        raw_text=raw_text,
        amount_chf=amount,
        date=date_match.group(1) if date_match else None,
        reference_number=ref_match.group(1).upper() if ref_match else None,
    )
