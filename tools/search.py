from __future__ import annotations
from loguru import logger
from duckduckgo_search import DDGS

BODY_TRUNCATE_LEN = 300

SEARCH_TRIGGERS = sorted([
    "search for", "look up", "find out", "what is the latest",
    "current news", "who is", "what happened to", "price of",
    "weather in", "how much does",
], key=len, reverse=True)


def extract_query(user_input: str) -> str:
    lower = user_input.lower()
    for trigger in SEARCH_TRIGGERS:
        idx = lower.find(trigger)
        if idx != -1:
            return user_input[idx + len(trigger):].strip()
    return user_input.strip()


def search(query: str, max_results: int = 3) -> list[dict]:
    logger.info(f"Searching: '{query}'")
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        logger.info(f"Got {len(results)} results")
        return results
    except Exception as exc:
        logger.error(f"Search error: {exc}")
        return []


def format_for_prompt(results: list[dict]) -> str:
    if not results:
        return "No search results found."
    lines = []
    for i, r in enumerate(results, 1):
        title = r.get("title", "")
        body = r.get("body", "")[:BODY_TRUNCATE_LEN]
        href = r.get("href", "")
        lines.append(f"{i}. {title}\n   {body}\n   Source: {href}")
    return "\n\n".join(lines)


if __name__ == "__main__":
    results = search("Python 3.13 new features")
    logger.info(format_for_prompt(results))
    query = extract_query("search for the best Python tutorials")
    assert query == "the best Python tutorials", f"Unexpected: '{query}'"
    logger.success("search.py smoke test passed")
