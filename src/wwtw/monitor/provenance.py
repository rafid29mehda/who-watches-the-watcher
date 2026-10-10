"""Deterministic argument provenance: where did each argument value of a proposed call come from?

Computed by code, so attacker-controlled text never has to be shown to the monitor verbatim.
"""

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from wwtw.entities import extract_entities, normalize_entity

USER = "USER"
_WS = re.compile(r"\s+")
_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}\b)")
_WORD = re.compile(r"\w+")


@dataclass(frozen=True)
class EntityProvenance:
    kind: str
    value: str
    sources: tuple[str, ...]


@dataclass(frozen=True)
class LeafProvenance:
    path: str
    kind: str  # "trivial" | "short" | "long"
    text: str
    display: str
    sources: tuple[str, ...] = ()
    overlap: tuple[tuple[str, float], ...] = ()
    entities: tuple[EntityProvenance, ...] = ()

    @property
    def user_grounded(self) -> bool:
        return USER in self.sources


def normalize(text: str) -> str:
    return _WS.sub(" ", text.casefold()).strip()


def canonical_number(x: int | float) -> str:
    f = float(x)
    return str(int(f)) if f.is_integer() else repr(f)


def _leaf_text(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if v is None:
        return "null"
    if isinstance(v, int | float):
        return canonical_number(v)
    return str(v)


def _display(v: Any) -> str:
    return json.dumps(v, ensure_ascii=False) if isinstance(v, str) else _leaf_text(v)


def flatten_args(args: Mapping[str, Any], prefix: str = "") -> list[tuple[str, Any]]:
    out: list[tuple[str, Any]] = []
    for k, v in args.items():
        out.extend(_flatten(f"{prefix}.{k}" if prefix else str(k), v))
    return out


def _flatten(path: str, v: Any) -> list[tuple[str, Any]]:
    if isinstance(v, Mapping):
        return flatten_args(v, path)
    if isinstance(v, list | tuple):
        res: list[tuple[str, Any]] = []
        for i, item in enumerate(v):
            res.extend(_flatten(f"{path}[{i}]", item))
        return res
    return [(path, v)]


def _number_re(canon: str) -> re.Pattern:
    tail = r"0*" if "." in canon else r"(?:\.0+)?"
    return re.compile(rf"(?<![\d.]){re.escape(canon)}{tail}(?!\d)(?!\.\d)")


def _shingles(text_norm: str, n: int = 5) -> set[tuple[str, ...]]:
    words = _WORD.findall(text_norm)
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def compute_provenance(
    user_prompt: str,
    tool_outputs: Sequence[tuple[str, int, str]],
    args: Mapping[str, Any],
    long_threshold: int = 120,
) -> list[LeafProvenance]:
    sources = [(USER, normalize(user_prompt))] + [
        (f"TOOL({fn}#{k})", normalize(text)) for fn, k, text in tool_outputs
    ]
    numeric_sources = [(label, _THOUSANDS.sub("", text)) for label, text in sources]
    shingle_cache: dict[str, set] = {}
    out: list[LeafProvenance] = []
    for path, raw in flatten_args(args):
        text = _leaf_text(raw)
        norm = normalize(text)
        if raw is None or isinstance(raw, bool) or len(norm) < 3:
            out.append(LeafProvenance(path, "trivial", text, _display(raw)))
        elif isinstance(raw, int | float):
            pat = _number_re(norm)
            labels = tuple(lbl for lbl, src in numeric_sources if pat.search(src))
            out.append(LeafProvenance(path, "short", text, _display(raw), labels))
        elif len(text) <= long_threshold:
            labels = tuple(lbl for lbl, src in sources if norm in src)
            out.append(LeafProvenance(path, "short", text, _display(raw), labels))
        else:
            leaf_sh = _shingles(norm)
            overlap = []
            for lbl, src in sources:
                if leaf_sh:
                    src_sh = shingle_cache.setdefault(lbl, _shingles(src))
                    frac = len(leaf_sh & src_sh) / len(leaf_sh)
                else:
                    frac = 1.0 if norm in src else 0.0
                if frac > 0:
                    overlap.append((lbl, frac))
            overlap.sort(key=lambda x: (-x[1], x[0]))
            ents = tuple(
                EntityProvenance(
                    kind,
                    value,
                    tuple(lbl for lbl, src in sources if normalize_entity(value) in src),
                )
                for kind, value in extract_entities(text)
            )
            out.append(LeafProvenance(path, "long", text, _display(raw), (), tuple(overlap), ents))
    return out
