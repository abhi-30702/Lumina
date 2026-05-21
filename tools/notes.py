from __future__ import annotations
from datetime import datetime
from pathlib import Path
from loguru import logger
from config import config


def _notes_dir() -> Path:
    d = config.resolved_notes_path()
    d.mkdir(parents=True, exist_ok=True)
    return d


def _filename(title: str | None = None) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if title:
        slug = title.lower().replace(" ", "_")[:40]
        return f"note_{ts}_{slug}.md"
    return f"note_{ts}.md"


def save_note(content: str, title: str | None = None) -> str:
    path = _notes_dir() / _filename(title)
    path.write_text(content, encoding="utf-8")
    logger.info(f"Saved note: {path.name}")
    return f"Note saved as {path.name}."


def append_note(content: str) -> str:
    notes = sorted(_notes_dir().glob("*.md"))
    if not notes:
        return save_note(content)
    latest = notes[-1]
    with open(latest, "a", encoding="utf-8") as f:
        f.write(f"\n{content}")
    logger.info(f"Appended to {latest.name}")
    return f"Added to {latest.name}."


def read_note(topic: str | None = None) -> str:
    notes = sorted(_notes_dir().glob("*.md"))
    if not notes:
        return "You have no saved notes."
    if topic:
        from fuzzywuzzy import process
        names = [n.name for n in notes]
        best, score = process.extractOne(topic, names)
        if score < 50:
            return f"No note found matching '{topic}'."
        target = _notes_dir() / best
        return target.read_text(encoding="utf-8")
    return "\n\n---\n\n".join(n.read_text(encoding="utf-8") for n in notes)


def list_notes() -> list[str]:
    return [n.name for n in sorted(_notes_dir().glob("*.md"))]


def delete_note(topic: str) -> str:
    notes = sorted(_notes_dir().glob("*.md"))
    if not notes:
        return "No notes to delete."
    from fuzzywuzzy import process
    names = [n.name for n in notes]
    best, score = process.extractOne(topic, names)
    if score < 50:
        return f"No note found matching '{topic}'."
    (_notes_dir() / best).unlink()
    logger.info(f"Deleted note: {best}")
    return f"Deleted note {best}."


if __name__ == "__main__":
    result = save_note("Buy milk, eggs, and bread.", title="groceries")
    logger.info(result)
    result = append_note("Also get coffee.")
    logger.info(result)
    notes = list_notes()
    logger.info(f"Notes: {notes}")
    content = read_note("groceries")
    logger.info(f"Content:\n{content}")
    result = delete_note("groceries")
    logger.info(result)
    logger.success("notes.py smoke test passed")
