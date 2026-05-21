from __future__ import annotations
import re
from datetime import datetime
from pathlib import Path
from loguru import logger
from config import config

SLUG_MAX_LEN = 40
FUZZY_MATCH_THRESHOLD = 50


def _notes_dir() -> Path:
    d = config.resolved_notes_path()
    d.mkdir(parents=True, exist_ok=True)
    return d


def _filename(title: str | None = None) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if title:
        slug = re.sub(r'[<>:"/\\|?*]', "", title.lower()).replace(" ", "_")[:SLUG_MAX_LEN] or "untitled"
        return f"note_{ts}_{slug}.md"
    return f"note_{ts}.md"


def _slug_of(filename: str) -> str:
    """Extract the slug portion of a note filename like 'note_<date>_<time>_<slug>.md'."""
    stem = Path(filename).stem
    parts = stem.split("_", 3)
    return parts[3] if len(parts) >= 4 else stem


def _match_note(topic: str, notes: list[Path]) -> Path | None:
    """Fuzzy-match a topic against note slugs. Returns the matching Path or None."""
    from fuzzywuzzy import process
    slug_map: dict[str, Path] = {}
    for n in notes:
        slug_map.setdefault(_slug_of(n.name), n)
    result = process.extractOne(topic, list(slug_map.keys()))
    if result is None or result[1] < FUZZY_MATCH_THRESHOLD:
        return None
    return slug_map[result[0]]


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
        target = _match_note(topic, notes)
        if target is None:
            return f"No note found matching '{topic}'."
        return target.read_text(encoding="utf-8")
    return "\n\n---\n\n".join(n.read_text(encoding="utf-8") for n in notes)


def list_notes() -> list[str]:
    return [n.name for n in sorted(_notes_dir().glob("*.md"))]


def delete_note(topic: str) -> str:
    notes = sorted(_notes_dir().glob("*.md"))
    if not notes:
        return "No notes to delete."
    target = _match_note(topic, notes)
    if target is None:
        return f"No note found matching '{topic}'."
    name = target.name
    target.unlink()
    logger.info(f"Deleted note: {name}")
    return f"Deleted note {name}."


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
