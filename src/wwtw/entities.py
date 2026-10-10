"""Extract emails, URLs/domains and IBANs from free text (shared by replay and provenance)."""

import re

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
URL_RE = re.compile(
    r"(?:https?://[^\s'\"<>]+|www\.[^\s'\"<>]+|"
    r"\b[a-z0-9-]+(?:\.[a-z0-9-]+)*\.(?:com|org|net|io|ai|edu|gov|info|biz|co|uk|de)\b"
    r"(?:/[^\s'\"<>]*)?)",
    re.IGNORECASE,
)
IBAN_RE = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b")


def normalize_entity(value: str) -> str:
    return value.casefold().strip().rstrip(".,;:!?)'\"")


def extract_entities(text: str) -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    seen: set[str] = set()

    def add(kind: str, value: str) -> None:
        key = normalize_entity(value)
        if key and key not in seen:
            seen.add(key)
            found.append((kind, value))

    for m in EMAIL_RE.finditer(text):
        add("email", m.group(0))
    without_emails = EMAIL_RE.sub(" ", text)
    for m in URL_RE.finditer(without_emails):
        add("url", m.group(0))
    for m in IBAN_RE.finditer(text):
        add("iban", m.group(0))
    return found


def attacker_entities(goal: str) -> frozenset[str]:
    return frozenset(normalize_entity(v) for _, v in extract_entities(goal))
