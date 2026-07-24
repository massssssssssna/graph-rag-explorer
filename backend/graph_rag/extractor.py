"""
backend/graph_rag/extractor.py
Extracts (subject, relation, object) triples from raw text using Groq.
Returns a list of Triple dataclass instances.
"""
import json
import logging
import re
from dataclasses import dataclass
from typing import List

from backend.llm import groq_client

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a precise knowledge-graph extractor.
Your only job is to read text and output Subject-Relation-Object triples.

Rules:
1. Output ONLY a valid JSON array — no markdown, no explanation, no extra text.
2. Each element must be: {"subject": "...", "relation": "...", "object": "..."}
3. Use short, lowercase, underscore-separated relation labels (e.g. "born_in", "founded_by", "located_in").
4. Subject and object should be proper nouns or named entities (people, places, organisations, concepts).
5. Extract as many triples as the text supports. If there are none, return [].
"""

EXTRACT_PROMPT = """Extract all knowledge triples from the following text:

\"\"\"
{text}
\"\"\"

Return ONLY the JSON array."""


@dataclass
class Triple:
    subject: str
    relation: str
    object: str

    def to_dict(self) -> dict:
        return {"subject": self.subject, "relation": self.relation, "object": self.object}

    def to_sentence(self) -> str:
        return f"{self.subject} {self.relation.replace('_', ' ')} {self.object}"


def _parse_triples(raw: str) -> List[Triple]:
    """
    Parse the JSON array returned by Groq.
    Falls back gracefully if the model returns wrapped markdown.
    """
    # Strip ```json ... ``` fences if present
    raw = raw.strip()
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if not match:
        logger.warning("No JSON array found in extractor response.")
        return []

    try:
        items = json.loads(match.group())
    except json.JSONDecodeError as exc:
        logger.error("JSON parse error in triple extraction: %s", exc)
        return []

    triples = []
    for item in items:
        try:
            t = Triple(
                subject=str(item["subject"]).strip().lower(),
                relation=str(item["relation"]).strip().lower().replace(" ", "_"),
                object=str(item["object"]).strip().lower(),
            )
            if t.subject and t.relation and t.object:
                triples.append(t)
        except (KeyError, TypeError):
            logger.debug("Skipping malformed triple: %s", item)

    return triples


def extract_triples(text: str) -> List[Triple]:
    """
    Send `text` to Groq and return extracted triples.
    Returns an empty list on any failure so the pipeline stays running.
    """
    if not text.strip():
        return []

    try:
        raw = groq_client.chat(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=EXTRACT_PROMPT.format(text=text),
        )
        triples = _parse_triples(raw)
        logger.info("Extracted %d triples from %d-char chunk.", len(triples), len(text))
        return triples
    except Exception as exc:
        logger.error("Triple extraction failed: %s", exc)
        return []
