from __future__ import annotations

import re
from collections import Counter
from typing import Iterable, List, Sequence, Tuple


STOPWORDS = {
    "a",
    "al",
    "and",
    "como",
    "con",
    "de",
    "del",
    "el",
    "en",
    "es",
    "esto",
    "for",
    "i",
    "la",
    "las",
    "lo",
    "los",
    "me",
    "mi",
    "my",
    "no",
    "of",
    "para",
    "por",
    "que",
    "the",
    "to",
    "un",
    "una",
    "y",
}

KNOWLEDGE_HINTS = ("aprend", "discover", "descubr", "means", "significa", "detecta", "detects", "prefiero")
CORRECTION_HINTS = ("actually", "correccion", "correct", "en realidad", "mejor dicho", "not", "no era")


def normalize_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def tokenize_text(text: str) -> List[str]:
    return re.findall(r"[\wáéíóúñüÁÉÍÓÚÑÜ]+", text.lower())


def hashed_token_ids(tokens: Sequence[str], vocab_size: int) -> List[int]:
    return [abs(hash(token)) % vocab_size for token in tokens]


def detect_intent(text: str) -> str:
    lowered = text.lower()
    if any(marker in lowered for marker in CORRECTION_HINTS):
        return "correction"
    if "?" in text:
        return "question"
    if "prefiero" in lowered or "prefer" in lowered:
        return "preference"
    if any(marker in lowered for marker in KNOWLEDGE_HINTS):
        return "knowledge_share"
    if len(lowered.split()) <= 3:
        return "small_talk"
    return "statement"


def extract_entities(text: str, tokens: Sequence[str]) -> List[str]:
    raw_entities = re.findall(r"\b[A-Z][A-Za-z0-9_]+\b", text)
    technical = [token for token in tokens if any(char.isdigit() for char in token) or "_" in token]
    entities = list(dict.fromkeys(raw_entities + technical))
    return entities[:8]


def infer_concepts(tokens: Sequence[str], entities: Sequence[str], top_k: int = 6) -> List[Tuple[str, float]]:
    filtered = [token for token in tokens if len(token) > 2 and token not in STOPWORDS]
    counts = Counter(filtered)
    total = sum(counts.values()) or 1
    concepts = [(token, count / total) for token, count in counts.most_common(top_k)]
    for entity in entities:
        if entity.lower() not in {name for name, _ in concepts}:
            concepts.append((entity, 0.8))
    return concepts[:top_k]


def compress_context(intent: str, concepts: Sequence[str], tokens: Sequence[str]) -> str:
    if concepts:
        joined = ", ".join(concepts[:4])
        return f"{intent}: {joined}"
    return f"{intent}: {' '.join(tokens[:12])}"


def concept_edges(concepts: Sequence[str]) -> List[Tuple[str, str, float]]:
    edges: List[Tuple[str, str, float]] = []
    for index, source in enumerate(concepts):
        for target in concepts[index + 1 : index + 3]:
            edges.append((source, target, max(0.15, 1.0 - 0.2 * index)))
    return edges
