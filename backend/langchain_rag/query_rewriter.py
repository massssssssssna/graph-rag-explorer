"""
backend/langchain_rag/query_rewriter.py
Fast query rewriting using a minimal Groq call.
- Uses llama-3.1-8b-instant (fastest model)
- max_tokens=120 (just 3 short strings needed)
- max_retries=1 (fail fast, no long backoff)
- Hard 6-second wall-clock timeout via thread
"""
import logging
import json
import re
import threading
from typing import List

logger = logging.getLogger(__name__)


def rewrite_query(question: str, timeout: float = 6.0) -> List[str]:
    """
    Generates up to 3 query variations.
    Returns [original_question] immediately on timeout or any error.
    """
    result = [question]

    def _run():
        try:
            import config
            from groq import Groq
            client = Groq(api_key=config.GROQ_API_KEY)
            resp = client.chat.completions.create(
                model="llama-3.1-8b-instant",   # fastest Groq model
                messages=[
                    {"role": "system", "content": "Output a JSON array of 3 search query variations. No explanation."},
                    {"role": "user", "content": (
                        f'Generate 3 different search queries for: "{question}"\n'
                        f'Return ONLY: ["query 1", "query 2", "query 3"]'
                    )},
                ],
                temperature=0.3,
                max_tokens=120,
            )
            raw = resp.choices[0].message.content or ""
            m = re.search(r"\[.*\]", raw, re.DOTALL)
            if m:
                queries = json.loads(m.group(0))
                if isinstance(queries, list):
                    extras = [str(q).strip() for q in queries if str(q).strip() and q != question]
                    result.extend(extras[:2])  # max 2 additional variations
        except Exception as exc:
            logger.warning("Query rewriter failed (skipping): %s", exc)

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout=timeout)
    if t.is_alive():
        logger.warning("Query rewriter timed out after %.0fs — using original query only.", timeout)

    logger.info("Query variations: %d for '%s'", len(result), question)
    return result
