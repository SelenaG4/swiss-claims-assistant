from app.pii import redact


def test_redacts_valid_ahv_number():
    r = redact("AHV-Nr: 756.1234.5678.97")
    assert r.entity_counts.get("AHV_NUMBER") == 1
    assert "756.1234.5678.97" not in r.redacted_text


def test_does_not_redact_invalid_ahv_checksum():
    # Same shape, wrong check digit -- shouldn't be treated as a real AHV number.
    r = redact("Reference: 756.1234.5678.00")
    assert "AHV_NUMBER" not in r.entity_counts


def test_redacts_swiss_iban():
    r = redact("IBAN: CH93 0076 2011 6238 5295 7")
    assert r.entity_counts.get("IBAN") == 1
    assert "6238" not in r.redacted_text


def test_redacts_email_and_phone():
    r = redact("Kontakt: matthias.keller@example.ch, Tel. +41 79 123 45 67")
    assert r.entity_counts.get("EMAIL") == 1
    assert r.entity_counts.get("PHONE") == 1


def test_redacts_date_of_birth_only_near_keyword():
    r = redact("geboren am 12.03.1985, Schadendatum: 01.06.2026")
    assert r.entity_counts.get("DATE_OF_BIRTH") == 1
    # the claim date is NOT a birth date and should survive redaction
    assert "01.06.2026" in r.redacted_text


def test_redacts_titled_names_in_three_languages():
    text = "Herr Matthias Keller / Monsieur Jean Dupont / Mrs. Anna Fischer"
    r = redact(text)
    assert r.entity_counts.get("PERSON_NAME") == 3


def test_leaves_non_pii_amounts_untouched():
    r = redact("Schadensumme: CHF 850.00, Police Nr. POL-2026-00193")
    assert "CHF 850.00" in r.redacted_text
    assert "POL-2026-00193" in r.redacted_text
    assert r.total_redactions == 0
