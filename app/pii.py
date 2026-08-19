"""Swiss-specific PII redaction -- the first stage in the pipeline, run before
any text reaches an external LLM/translation API, in line with the nFADP
(revised Swiss Federal Act on Data Protection) and FINMA expectations that
Swiss enterprises govern PII before it leaves their systems.

Deliberately rule-based, not a full NER model: it reliably catches the
highest-value, highest-risk *structured* PII in Swiss claims documents --
AHV/AVS numbers, Swiss IBANs, phone numbers, postal addresses, dates of
birth -- which is exactly the PII with the clearest legal exposure if leaked
(an AHV number alone is enough to link a document to a specific person under
Swiss law). Free-text personal names are caught with a title-anchored
heuristic (Herr/Frau/M./Mme/...), which is intentionally narrower than a real
NER model would be -- see "What this doesn't catch" in the README.

Nothing here calls out to a network. It's the layer that exists specifically
so nothing has to.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# AHV/AVS number: 756.XXXX.XXXX.XX (Switzerland's country prefix is always 756).
# Real numbers carry an EAN-13 check digit; we validate it so we don't redact
# something that merely *looks* like an AHV number (e.g. an invoice line ID).
_AHV_RE = re.compile(r"\b756\.\d{4}\.\d{4}\.\d{2}\b")

# Swiss IBAN: CH or LI + 2 check digits + 5-digit bank clearing number + 12
# alphanumeric account identifier = 21 characters total, conventionally
# displayed in space-separated blocks of 4 (final block just 1 character).
_IBAN_RE = re.compile(r"\b(?:CH|LI)\d{2}(?:[ ]?[0-9A-Z]{4}){4}[ ]?[0-9A-Z]\b")

_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")

# Swiss phone: +41 79 123 45 67, 079 123 45 67, 0041 79 123 45 67, etc.
_PHONE_RE = re.compile(r"(?:\+41|0041|0)\s?\d{2}[\s.]?\d{3}[\s.]?\d{2}[\s.]?\d{2}\b")

# 4-digit Swiss postal code immediately followed by a capitalized place name.
_POSTAL_RE = re.compile(r"\b\d{4}\s+[A-ZÄÖÜ][a-zäöüéèà]+(?:[- ][A-ZÄÖÜ][a-zäöüéèà]+)*\b")

# Dates of birth: only redact a date immediately preceded by a birth-date
# keyword, in DE/FR/EN, so we don't blanket-redact every date in a document
# (a claim date or invoice date is not PII in the same sense).
_DOB_RE = re.compile(
    r"(?:geboren am|geburtsdatum|né\(e\) le|née? le|date de naissance|date of birth|born on|dob)"
    r"\s*[:\-]?\s*(\d{1,2}[./]\d{1,2}[./]\d{2,4})",
    re.IGNORECASE,
)

# Personal names: a title/honorific in DE/FR/EN followed by 1-2 capitalized
# words. Narrower than real NER on purpose -- see module docstring.
_NAME_RE = re.compile(
    r"\b(?:Herr|Frau|Monsieur|Madame|Mme|M\.|Mr\.|Mrs\.|Ms\.)\s+"
    r"([A-ZÄÖÜ][a-zäöüéèà]+(?:[- ][A-ZÄÖÜ][a-zäöüéèà]+){0,2})"
)

AHV_CHECK_WEIGHTS = [1, 3] * 6  # EAN-13 style alternating weights for the 12 payload digits


def _ahv_checksum_valid(match: str) -> bool:
    digits = match.replace(".", "")
    if len(digits) != 13:
        return False
    payload, check = digits[:12], int(digits[12])
    total = sum(int(d) * w for d, w in zip(payload, AHV_CHECK_WEIGHTS))
    computed = (10 - (total % 10)) % 10
    return computed == check


@dataclass
class RedactionResult:
    redacted_text: str
    # Audit trail: counts by entity type only -- never the raw matched value,
    # so the audit log itself can't leak what it's proving was redacted.
    entity_counts: dict[str, int] = field(default_factory=dict)

    @property
    def total_redactions(self) -> int:
        return sum(self.entity_counts.values())


def redact(text: str) -> RedactionResult:
    counts: dict[str, int] = {}

    def _sub(pattern: re.Pattern, label: str, text: str, validate=None) -> str:
        def repl(m: re.Match) -> str:
            if validate is not None and not validate(m.group(0)):
                return m.group(0)
            counts[label] = counts.get(label, 0) + 1
            return f"[REDACTED:{label}]"

        return pattern.sub(repl, text)

    text = _sub(_AHV_RE, "AHV_NUMBER", text, validate=_ahv_checksum_valid)
    text = _sub(_IBAN_RE, "IBAN", text)
    text = _sub(_EMAIL_RE, "EMAIL", text)
    text = _sub(_DOB_RE, "DATE_OF_BIRTH", text)  # must run before generic phone/postal patterns
    text = _sub(_PHONE_RE, "PHONE", text)
    text = _sub(_NAME_RE, "PERSON_NAME", text)
    text = _sub(_POSTAL_RE, "POSTAL_ADDRESS", text)

    return RedactionResult(redacted_text=text, entity_counts=counts)
