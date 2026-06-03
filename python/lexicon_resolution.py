"""Gloss alias and fallback resolution for the WLASL pose lexicon.

The pose generator already falls back to fingerspelling for unknown words.
This module keeps that behavior, but gives common vocabulary variants a cheap
chance to resolve to real WLASL signs first.
"""

from __future__ import annotations

import csv
import json
import os
import re
from dataclasses import dataclass
from difflib import get_close_matches
from pathlib import Path
from typing import Iterable


TOKEN_RE = re.compile(r"[a-zA-Z0-9_']+")


def normalize_key(value: str) -> str:
    """Normalize user text for alias lookup while preserving word boundaries."""
    value = value.strip().lower().replace("&", " and ")
    value = re.sub(r"[_-]+", " ", value)
    value = re.sub(r"[^a-z0-9']+", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def gloss_slug(value: str) -> str:
    """Normalize text into the underscore form used by WLASL gloss paths."""
    key = normalize_key(value)
    key = key.replace("'", "")
    return key.replace(" ", "_")


def _alias_target(value) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        target = value.get("target") or value.get("gloss") or value.get("maps_to")
        return str(target) if target else None
    return None


def load_aliases(alias_path: str | os.PathLike | None) -> dict[str, str]:
    """Load aliases from JSON.

    Accepted shapes:
      {"aliases": {"mom": "mother", "thank you": "thank_you"}}
      {"mom": "mother"}

    Values may also be objects with a "target", "gloss", or "maps_to" field.
    """
    if not alias_path:
        return {}
    path = Path(alias_path)
    if not path.is_file():
        return {}

    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Alias file must contain a JSON object: {path}")

    aliases = raw.get("aliases", raw)
    if not isinstance(aliases, dict):
        raise ValueError(f"Alias file 'aliases' must be a JSON object: {path}")

    normalized: dict[str, str] = {}
    for source, target_value in aliases.items():
        target = _alias_target(target_value)
        if not target:
            continue
        source_key = normalize_key(str(source))
        target_slug = gloss_slug(target)
        if source_key and target_slug:
            normalized[source_key] = target_slug
    return normalized


def load_lexicon_glosses(lexicon_dir: str | os.PathLike) -> set[str]:
    index_path = Path(lexicon_dir) / "index.csv"
    if not index_path.is_file():
        return set()

    glosses: set[str] = set()
    with index_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            for field in ("glosses", "words"):
                value = (row.get(field) or "").strip()
                if value:
                    glosses.add(gloss_slug(value))
    return glosses


def variant_candidates(value: str) -> list[str]:
    """Cheap English variants that commonly block lexicon hits."""
    slug = gloss_slug(value)
    if not slug:
        return []

    variants = [slug]
    compact = slug.replace("_", "")
    if compact != slug:
        variants.append(compact)

    if slug.endswith("ies") and len(slug) > 4:
        variants.append(slug[:-3] + "y")
    if slug.endswith("es") and len(slug) > 3:
        variants.append(slug[:-2])
    if slug.endswith("s") and len(slug) > 3:
        variants.append(slug[:-1])
    if slug.endswith("ing") and len(slug) > 5:
        base = slug[:-3]
        variants.append(base)
        if len(base) > 2 and base[-1] == base[-2]:
            variants.append(base[:-1])
        variants.append(base + "e")
    if slug.endswith("ed") and len(slug) > 4:
        base = slug[:-2]
        variants.append(base)
        if len(base) > 2 and base[-1] == base[-2]:
            variants.append(base[:-1])
        variants.append(base + "e")
    if slug.endswith("'s"):
        variants.append(slug[:-2])

    seen = set()
    out = []
    for variant in variants:
        if variant and variant not in seen:
            seen.add(variant)
            out.append(variant)
    return out


@dataclass(frozen=True)
class ResolvedGloss:
    source: str
    target: str
    method: str


class GlossResolver:
    """Resolve user/requested glosses before the fingerspelling fallback."""

    def __init__(self, lexicon_glosses: Iterable[str], aliases: dict[str, str] | None = None):
        self.lexicon_glosses = {gloss_slug(g) for g in lexicon_glosses if gloss_slug(g)}
        self.aliases = aliases or {}
        self.max_phrase_words = max(
            [1]
            + [len(key.split()) for key in self.aliases]
            + [len(gloss.split("_")) for gloss in self.lexicon_glosses]
        )

    @classmethod
    def from_paths(cls, lexicon_dir: str | os.PathLike, alias_path: str | os.PathLike | None = None):
        return cls(load_lexicon_glosses(lexicon_dir), load_aliases(alias_path))

    def resolve_term(self, value: str) -> ResolvedGloss:
        key = normalize_key(value)
        if not key:
            return ResolvedGloss(value, value, "empty")

        alias_target = self.aliases.get(key)
        if alias_target:
            target = self._resolve_slug(alias_target)
            return ResolvedGloss(value, target or alias_target, "alias" if target else "alias_unresolved")

        for candidate in variant_candidates(value):
            if candidate in self.lexicon_glosses:
                method = "exact" if candidate == gloss_slug(value) else "variant"
                return ResolvedGloss(value, candidate, method)

        return ResolvedGloss(value, gloss_slug(value), "unresolved")

    def resolve_text(self, text: str) -> tuple[str, list[ResolvedGloss]]:
        tokens = [match.group(0) for match in TOKEN_RE.finditer(text.lower())]
        if not tokens:
            return text, []

        resolved: list[ResolvedGloss] = []
        out: list[str] = []
        i = 0
        while i < len(tokens):
            best: tuple[int, ResolvedGloss] | None = None
            max_len = min(self.max_phrase_words, len(tokens) - i)
            for size in range(max_len, 0, -1):
                phrase = " ".join(tokens[i : i + size])
                item = self.resolve_term(phrase)
                if item.method != "unresolved" or size == 1:
                    best = (size, item)
                    break
            assert best is not None
            size, item = best
            resolved.append(item)
            out.append(item.target)
            i += size

        return " ".join(out), resolved

    def _resolve_slug(self, value: str) -> str | None:
        for candidate in variant_candidates(value):
            if candidate in self.lexicon_glosses:
                return candidate
        return None

    def unresolved_terms(self, terms: Iterable[str]) -> list[ResolvedGloss]:
        return [item for term in terms if (item := self.resolve_term(term)).method == "unresolved"]

    def suggest_mappings(self, term: str, limit: int = 5) -> list[str]:
        candidates = [candidate for candidate in variant_candidates(term) if candidate in self.lexicon_glosses]
        if candidates:
            return candidates[:limit]
        slug = gloss_slug(term)
        if not slug:
            return []
        return get_close_matches(slug, sorted(self.lexicon_glosses), n=limit, cutoff=0.72)

