from __future__ import annotations
import re
from loguru import logger

SEARCH_TRIGGERS = [
    "search for", "look up", "find out", "what is the latest",
    "current news", "who is", "what happened to", "price of",
    "weather in", "how much does",
]
NOTE_SAVE_TRIGGERS = ["save a note", "write this down", "add to my notes"]
NOTE_READ_TRIGGERS = ["read my notes", "what did i save", "read note about", "list my notes"]
NOTE_DELETE_TRIGGERS = ["delete note about"]
MEMORY_SAVE_TRIGGERS = ["remember that"]
MEMORY_FORGET_TRIGGERS = ["forget tag", "forget the fact"]
SEARCH_ROUTE = "search"
NOTE_SAVE_ROUTE = "note_save"
NOTE_READ_ROUTE = "note_read"
NOTE_DELETE_ROUTE = "note_delete"
MEMORY_SAVE_ROUTE = "memory_save"
MEMORY_FORGET_ROUTE = "memory_forget"
LLM_ROUTE = "llm"


def classify(text: str) -> tuple[str, str]:
    """Return (route, extracted_arg)."""
    lower = text.lower().strip()

    for trigger in MEMORY_SAVE_TRIGGERS:
        if trigger in lower:
            arg = re.split(re.escape(trigger), lower, maxsplit=1, flags=re.IGNORECASE)[-1].strip()
            return MEMORY_SAVE_ROUTE, arg

    for trigger in MEMORY_FORGET_TRIGGERS:
        if lower.startswith(trigger):
            arg = lower[len(trigger):].strip()
            return MEMORY_FORGET_ROUTE, arg

    for trigger in NOTE_DELETE_TRIGGERS:
        if trigger in lower:
            arg = re.split(re.escape(trigger), lower, maxsplit=1, flags=re.IGNORECASE)[-1].strip()
            return NOTE_DELETE_ROUTE, arg

    for trigger in NOTE_SAVE_TRIGGERS:
        if trigger in lower:
            parts = re.split(r":\s*", text, maxsplit=1)
            if len(parts) > 1:
                arg = parts[1].strip()
            else:
                # No colon — take everything after the trigger phrase
                idx = lower.index(trigger) + len(trigger)
                arg = text[idx:].strip()
            return NOTE_SAVE_ROUTE, arg

    for trigger in NOTE_READ_TRIGGERS:
        if trigger in lower:
            arg = ""
            if "about" in lower:
                arg = lower.split("about", 1)[-1].strip()
            return NOTE_READ_ROUTE, arg

    for trigger in SEARCH_TRIGGERS:
        if trigger in lower:
            arg = lower.split(trigger, 1)[-1].strip()
            return SEARCH_ROUTE, arg

    return LLM_ROUTE, text


if __name__ == "__main__":
    cases = [
        ("search for the latest python news", SEARCH_ROUTE),
        ("remember that my name is Alex", MEMORY_SAVE_ROUTE),
        ("forget tag name", MEMORY_FORGET_ROUTE),
        ("save a note: buy groceries", NOTE_SAVE_ROUTE),
        ("read my notes", NOTE_READ_ROUTE),
        ("what time is it", LLM_ROUTE),
        ("save a note buy milk", NOTE_SAVE_ROUTE),
        ("forget my password", LLM_ROUTE),
    ]
    all_ok = True
    for text, expected_route in cases:
        route, arg = classify(text)
        status = "OK" if route == expected_route else "FAIL"
        if status == "FAIL":
            all_ok = False
        logger.info(f"[{status}] '{text}' → {route} (arg='{arg}')")
    assert all_ok, "Some routing cases failed"
    logger.success("router.py smoke test passed")
