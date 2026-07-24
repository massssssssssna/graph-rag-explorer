"""
backend/llm/groq_client.py
Thin wrapper around the Groq Python SDK.
Provides a single `chat()` function with exponential-backoff retry.
"""
import time
import logging
from groq import Groq, RateLimitError, APIStatusError
import config

logger = logging.getLogger(__name__)

# One shared client instance (thread-safe for reads)
_client: Groq | None = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        if not config.GROQ_API_KEY:
            raise ValueError(
                "GROQ_API_KEY is not set. "
                "Copy .env.example to .env and fill in your key."
            )
        _client = Groq(api_key=config.GROQ_API_KEY)
    return _client


def chat(
    system_prompt: str,
    user_prompt: str,
    model: str = config.GROQ_MODEL,
    temperature: float = config.GROQ_TEMPERATURE,
    max_tokens: int = config.GROQ_MAX_TOKENS,
    max_retries: int = 3,
) -> str:
    """
    Send a chat completion request to Groq and return the response text.

    Retries up to `max_retries` times on rate-limit errors with exponential
    backoff (2s, 4s, 8s …).

    Raises:
        ValueError: if GROQ_API_KEY is missing.
        RuntimeError: if all retries are exhausted.
    """
    client = _get_client()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ]

    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            text = response.choices[0].message.content or ""
            logger.debug("Groq responded (%d chars) on attempt %d", len(text), attempt)
            return text.strip()

        except RateLimitError:
            wait = 2 ** attempt
            logger.warning("Groq rate-limited. Retrying in %ds (attempt %d/%d)…",
                           wait, attempt, max_retries)
            time.sleep(wait)

        except APIStatusError as exc:
            logger.error("Groq API error: %s", exc)
            raise

    raise RuntimeError(f"Groq request failed after {max_retries} retries (rate limit).")
