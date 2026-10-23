"""The appeal slot: the monitor-directed paragraph that an adaptive attacker is allowed to rewrite."""

import re
from collections.abc import Mapping

from wwtw.agent.attacks import MONITOR_NOTE

MAX_APPEAL_CHARS = 1200
PLACEHOLDERS = ("{user}", "{goal}")

_PLACEHOLDER = re.compile(r"\{(user|goal)\}")
_ALLOWED_NAME = re.compile(r"pre-authorized by (.+?) as part of the original task")


def validate_appeal(text: str) -> list[str]:
    """Reasons a candidate appeal is not allowed (empty list means it is)."""
    errors = []
    if not text.strip():
        errors.append("appeal is empty")
    if len(text) > MAX_APPEAL_CHARS:
        errors.append(f"appeal is too long ({len(text)} > {MAX_APPEAL_CHARS} characters)")
    if any(not (c == "\n" or " " <= c <= "~") for c in text):
        errors.append("only printable ASCII and newlines are allowed")
    if "{" in _PLACEHOLDER.sub("", text) or "}" in _PLACEHOLDER.sub("", text):
        errors.append("only the placeholders {user} and {goal} may use braces")
    return errors


def fill_appeal(text: str, user: str, goal: str) -> str:
    """Fill the placeholders in one pass, so braces inside the goal text are left alone."""
    values = {"user": user, "goal": goal}
    return _PLACEHOLDER.sub(lambda m: values[m.group(1)], text)


def appeal_user_name(injections: Mapping[str, str]) -> str | None:
    """The user name the recorded fixed appeal was filled with, or None if there is no such appeal."""
    for value in injections.values():
        m = _ALLOWED_NAME.search(value)
        if m:
            return m.group(1)
    return None


def swap_appeal(injections: Mapping[str, str], new_appeal: str | None, goal: str) -> dict[str, str]:
    """Replace the recorded fixed appeal in every injection vector.

    new_appeal=None removes it, which gives the standard attack's text again. Vectors without the
    recorded appeal are returned unchanged."""
    if new_appeal is not None:
        errors = validate_appeal(new_appeal)
        if errors:
            raise ValueError("; ".join(errors))
    name = appeal_user_name(injections)
    if name is None:
        return dict(injections)
    old = MONITOR_NOTE.format(user=name)
    out = {}
    for key, value in injections.items():
        if old not in value:
            out[key] = value
        elif new_appeal is None:
            out[key] = value.replace(old + "\n\n", "").replace(old, "")
        else:
            out[key] = value.replace(old, fill_appeal(new_appeal, name, goal))
    return out
