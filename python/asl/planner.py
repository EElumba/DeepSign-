from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SignEntry:
    gloss: str
    english: tuple[str, ...]
    nonmanual: dict[str, Any] | None = None


PHRASES: dict[str, list[SignEntry]] = {
    "hello": [SignEntry("HELLO", ("hello",))],
    "hi": [SignEntry("HELLO", ("hi",))],
    "thank you": [SignEntry("THANK-YOU", ("thank", "you"))],
    "thanks": [SignEntry("THANK-YOU", ("thanks",))],
    "nice to meet you": [
        SignEntry("NICE", ("nice",)),
        SignEntry("MEET", ("meet",)),
        SignEntry("YOU", ("you",)),
    ],
    "i need help": [
        SignEntry("ME", ("i",)),
        SignEntry("NEED", ("need",)),
        SignEntry("HELP", ("help",)),
    ],
    "what is your name": [
        SignEntry("YOUR", ("your",)),
        SignEntry("NAME", ("name",)),
        SignEntry("WHAT", ("what",), {"brows": "furrowed", "head": "forward"}),
    ],
    "my name is": [
        SignEntry("MY", ("my",)),
        SignEntry("NAME", ("name",)),
    ],
}

LEXICON: dict[str, SignEntry] = {
    "i": SignEntry("ME", ("i",)),
    "me": SignEntry("ME", ("me",)),
    "my": SignEntry("MY", ("my",)),
    "you": SignEntry("YOU", ("you",)),
    "your": SignEntry("YOUR", ("your",)),
    "help": SignEntry("HELP", ("help",)),
    "need": SignEntry("NEED", ("need",)),
    "want": SignEntry("WANT", ("want",)),
    "go": SignEntry("GO", ("go",)),
    "home": SignEntry("HOME", ("home",)),
    "school": SignEntry("SCHOOL", ("school",)),
    "work": SignEntry("WORK", ("work",)),
    "learn": SignEntry("LEARN", ("learn",)),
    "sign": SignEntry("SIGN", ("sign",)),
    "asl": SignEntry("ASL", ("asl",)),
    "yes": SignEntry("YES", ("yes",)),
    "no": SignEntry("NO", ("no",)),
    "please": SignEntry("PLEASE", ("please",)),
    "sorry": SignEntry("SORRY", ("sorry",)),
    "name": SignEntry("NAME", ("name",)),
    "what": SignEntry("WHAT", ("what",), {"brows": "furrowed"}),
    "where": SignEntry("WHERE", ("where",), {"brows": "furrowed"}),
    "when": SignEntry("WHEN", ("when",), {"brows": "furrowed"}),
    "who": SignEntry("WHO", ("who",), {"brows": "furrowed"}),
    "why": SignEntry("WHY", ("why",), {"brows": "furrowed"}),
}


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9'#\s-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def plan_asl(text: str) -> dict[str, Any]:
    normalized = normalize_text(text)
    if not normalized:
        return {"mode": "empty", "sourceText": text, "units": []}

    if normalized in PHRASES:
        units = [_entry_to_unit(entry, "phrase") for entry in PHRASES[normalized]]
        return {
            "mode": "curated_phrase",
            "sourceText": text,
            "normalizedText": normalized,
            "units": units,
        }

    units: list[dict[str, Any]] = []
    for token in normalized.split():
      entry = LEXICON.get(token)
      if entry:
          units.append(_entry_to_unit(entry, "lexicon"))
      elif token.isalpha() and len(token) <= 24:
          units.append({
              "type": "fingerspell",
              "text": token,
              "letters": list(token.upper()),
              "reason": "not_in_curated_lexicon",
          })
      else:
          units.append({
              "type": "caption",
              "text": token,
              "reason": "unsupported_token",
          })

    mode = "mixed_lexicon_fingerspell"
    if all(unit["type"] == "sign" for unit in units):
        mode = "lexicon_sequence"
    elif all(unit["type"] == "fingerspell" for unit in units):
        mode = "fingerspell_only"

    return {
        "mode": mode,
        "sourceText": text,
        "normalizedText": normalized,
        "units": units,
    }


def _entry_to_unit(entry: SignEntry, source: str) -> dict[str, Any]:
    unit = {
        "type": "sign",
        "gloss": entry.gloss,
        "english": list(entry.english),
        "source": source,
    }
    if entry.nonmanual:
        unit["nonmanual"] = entry.nonmanual
    return unit
