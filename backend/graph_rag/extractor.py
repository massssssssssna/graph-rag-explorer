"""
backend/graph_rag/extractor.py
Extracts (subject, relation, object) triples from raw text using regex table parsing + Groq LLM extraction.
"""
import json
import logging
import re
from dataclasses import dataclass
from typing import List

from backend.llm import groq_client

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a precise knowledge-graph extractor.
Output Subject-Relation-Object triples as a JSON array:
[{"subject": "...", "relation": "...", "object": "..."}]

Rules:
1. Output ONLY a valid JSON array.
2. Subject and Object must be named entities (e.g. "NASA", "Mars", "Falcon 9", "OpenAI", "GPT-4").
3. Relation should be short, lowercase, underscore-separated (e.g. "launched_by", "orbits", "develops", "competes_with").
"""


@dataclass
class Triple:
    subject: str
    relation: str
    object: str

    def to_dict(self) -> dict:
        return {"subject": self.subject, "relation": self.relation, "object": self.object}

    def to_sentence(self) -> str:
        return f"{self.subject} {self.relation.replace('_', ' ')} {self.object}"


def _parse_table_triples(text: str) -> List[Triple]:
    """Parse pipe-delimited markdown triple tables (e.g. Subject | Relation | Object)."""
    triples = []
    lines = text.splitlines()
    for line in lines:
        if "|" in line:
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 3:
                subj = parts[0].strip()
                rel = parts[1].strip()
                obj = parts[2].strip()
                # Skip header rows
                if (subj.lower() in ("subject", "subject entity", "---", "") or
                    rel.lower() in ("relation", "relation (predicate)", "---", "") or
                    obj.lower() in ("object", "object entity", "---", "")):
                    continue
                if subj and rel and obj and not subj.startswith("-") and not rel.startswith("-"):
                    t = Triple(
                        subject=subj.lower(),
                        relation=rel.lower().replace(" ", "_"),
                        object=obj.lower(),
                    )
                    triples.append(t)
    return triples


def _parse_declarative_triples(text: str) -> List[Triple]:
    """Pattern matching for clear declarative sentences like 'Earth orbits Sun.' or 'SpaceX develops Falcon 9.'"""
    triples = []
    patterns = [
        r"([A-Z][a-zA-Z0-9\s_\-]+?)\s+(orbits|landed_on|launched_by|carries|explores|discovered|develops|releases|powers|invests_in|partners_with|founded_by|competes_with|operates)\s+([A-Z0-9][a-zA-Z0-9\s_\-]+?)(?:\.|\n|$)"
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text)
        for subj, rel, obj in matches:
            t = Triple(
                subject=subj.strip().lower(),
                relation=rel.strip().lower().replace(" ", "_"),
                object=obj.strip().lower(),
            )
            triples.append(t)
    return triples


def extract_triples(text: str) -> List[Triple]:
    """
    Extracts knowledge triples from text.
    First uses fast regex table/pattern parsing, then falls back to Groq if needed.
    """
    if not text or not text.strip():
        return []

    # 1. Fast table parse
    table_triples = _parse_table_triples(text)
    pattern_triples = _parse_declarative_triples(text)

    combined = table_triples + pattern_triples
    if len(combined) >= 5:
        logger.info("Parsed %d triples from text using table/pattern rules.", len(combined))
        return combined

    # 2. LLM fallback if text is freeform narrative
    try:
        raw = groq_client.chat(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=f"Extract knowledge triples from:\n\"\"\"\n{text[:2000]}\n\"\"\"",
            max_tokens=400,
        )
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if match:
            items = json.loads(match.group())
            for item in items:
                if isinstance(item, dict) and "subject" in item and "relation" in item and "object" in item:
                    t = Triple(
                        subject=str(item["subject"]).strip().lower(),
                        relation=str(item["relation"]).strip().lower().replace(" ", "_"),
                        object=str(item["object"]).strip().lower(),
                    )
                    if t.subject and t.relation and t.object:
                        combined.append(t)
    except Exception as exc:
        logger.warning("LLM triple extraction skipped: %s", exc)

    return combined
