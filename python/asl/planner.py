from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SignEntry:
    gloss: str
    english: tuple[str, ...]
    hands: dict[str, Any] | None = None
    nonmanual: dict[str, Any] | None = None


ONE_HANDED = {
    "pattern": "one_handed",
    "active": ["dominant"],
    "dominant": {"role": "articulator"},
    "nonDominant": {"role": "inactive"},
}

SYMMETRICAL = {
    "pattern": "symmetrical",
    "active": ["dominant", "non_dominant"],
    "dominant": {"role": "mirror_articulator"},
    "nonDominant": {"role": "mirror_articulator"},
    "mirror": True,
}

ALTERNATING_SYMMETRICAL = {
    "pattern": "symmetrical",
    "active": ["dominant", "non_dominant"],
    "dominant": {"role": "alternating_articulator", "phase": 0.0},
    "nonDominant": {"role": "alternating_articulator", "phase": 0.5},
    "mirror": True,
    "alternating": True,
}

ASYMMETRICAL_SUPPORT = {
    "pattern": "asymmetrical",
    "active": ["dominant", "non_dominant"],
    "dominant": {"role": "articulator"},
    "nonDominant": {"role": "support", "motion": "hold"},
    "mirror": False,
}

ASYMMETRICAL_TARGET = {
    "pattern": "asymmetrical",
    "active": ["dominant", "non_dominant"],
    "dominant": {"role": "articulator"},
    "nonDominant": {"role": "target", "motion": "hold"},
    "mirror": False,
}

FINGERSPELL_HANDS = {
    "pattern": "asymmetrical",
    "active": ["dominant", "non_dominant"],
    "dominant": {"role": "fingerspell"},
    "nonDominant": {"role": "reference", "motion": "hold"},
    "mirror": False,
}


PHRASES: dict[str, list[SignEntry]] = {
    "hello": [SignEntry("HELLO", ("hello",))],
    "hi": [SignEntry("HELLO", ("hi",))],
    "thank you": [SignEntry("THANK-YOU", ("thank", "you"))],
    "thanks": [SignEntry("THANK-YOU", ("thanks",))],
    "nice to meet you": [
        SignEntry("NICE", ("nice",)),
        SignEntry("MEET", ("meet",), SYMMETRICAL),
        SignEntry("YOU", ("you",)),
    ],
    "i need help": [
        SignEntry("ME", ("i",)),
        SignEntry("NEED", ("need",)),
        SignEntry("HELP", ("help",), ASYMMETRICAL_SUPPORT),
    ],
    "what is your name": [
        SignEntry("YOUR", ("your",)),
        SignEntry("NAME", ("name",), ASYMMETRICAL_TARGET),
        SignEntry("WHAT", ("what",), SYMMETRICAL, {"brows": "furrowed", "head": "forward"}),
    ],
    "my name is": [
        SignEntry("MY", ("my",)),
        SignEntry("NAME", ("name",), ASYMMETRICAL_TARGET),
    ],
    "i work at school": [
        SignEntry("ME", ("i",)),
        SignEntry("WORK", ("work",), SYMMETRICAL),
        SignEntry("SCHOOL", ("school",), SYMMETRICAL),
    ],
    "testing how are you doing today": [
        SignEntry("TEST", ("testing",), ASYMMETRICAL_TARGET),
        SignEntry("HOW", ("how",), SYMMETRICAL),
        SignEntry("YOU", ("you",)),
        SignEntry("DO", ("doing",), SYMMETRICAL),
        SignEntry("TODAY", ("today",), SYMMETRICAL),
    ],
}

LEXICON: dict[str, SignEntry] = {
    "i": SignEntry("ME", ("i",)),
    "me": SignEntry("ME", ("me",)),
    "my": SignEntry("MY", ("my",)),
    "you": SignEntry("YOU", ("you",)),
    "your": SignEntry("YOUR", ("your",)),
    "help": SignEntry("HELP", ("help",), ASYMMETRICAL_SUPPORT),
    "need": SignEntry("NEED", ("need",)),
    "want": SignEntry("WANT", ("want",)),
    "go": SignEntry("GO", ("go",)),
    "home": SignEntry("HOME", ("home",)),
    "school": SignEntry("SCHOOL", ("school",), SYMMETRICAL),
    "work": SignEntry("WORK", ("work",), SYMMETRICAL),
    "learn": SignEntry("LEARN", ("learn",), ASYMMETRICAL_TARGET),
    "sign": SignEntry("SIGN", ("sign",), ALTERNATING_SYMMETRICAL),
    "test": SignEntry("TEST", ("test",), ASYMMETRICAL_TARGET),
    "quiz": SignEntry("TEST", ("quiz",), ASYMMETRICAL_TARGET),
    "how": SignEntry("HOW", ("how",), SYMMETRICAL, {"brows": "furrowed"}),
    "do": SignEntry("DO", ("do",), SYMMETRICAL),
    "doing": SignEntry("DO", ("doing",), SYMMETRICAL),
    "today": SignEntry("TODAY", ("today",), SYMMETRICAL),
    "now": SignEntry("NOW", ("now",), SYMMETRICAL),
    "asl": SignEntry("ASL", ("asl",)),
    "yes": SignEntry("YES", ("yes",)),
    "no": SignEntry("NO", ("no",)),
    "please": SignEntry("PLEASE", ("please",)),
    "sorry": SignEntry("SORRY", ("sorry",)),
    "name": SignEntry("NAME", ("name",), ASYMMETRICAL_TARGET),
    "meet": SignEntry("MEET", ("meet",), SYMMETRICAL),
    "what": SignEntry("WHAT", ("what",), SYMMETRICAL, {"brows": "furrowed"}),
    "where": SignEntry("WHERE", ("where",), None, {"brows": "furrowed"}),
    "when": SignEntry("WHEN", ("when",), None, {"brows": "furrowed"}),
    "who": SignEntry("WHO", ("who",), None, {"brows": "furrowed"}),
    "why": SignEntry("WHY", ("why",), None, {"brows": "furrowed"}),
}

TOKEN_REWRITES = {
    "testing": "test",
    "tested": "test",
    "tests": "test",
    "doing": "do",
}

SKIP_WORDS = {"am", "is", "are", "was", "were", "be", "being", "been", "the", "a", "an"}


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
        if token in SKIP_WORDS:
            continue
        lookup = TOKEN_REWRITES.get(token, token)
        entry = LEXICON.get(lookup)
        if entry:
            units.append(_entry_to_unit(entry, "lexicon"))
        elif token.isalpha() and len(token) <= 24:
            units.append({
                "type": "fingerspell",
                "text": token,
                "letters": list(token.upper()),
                "hands": FINGERSPELL_HANDS,
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
        "hands": entry.hands or ONE_HANDED,
        "source": source,
    }
    if entry.nonmanual:
        unit["nonmanual"] = entry.nonmanual
    return unit
