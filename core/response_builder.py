from __future__ import annotations
from loguru import logger


def build_prompt(
    user_input: str,
    memory_chunks: list[str],
    web_context: str | None = None,
) -> str:
    """Assemble the final user-facing prompt string with injected context."""
    parts: list[str] = []

    if memory_chunks:
        mem_lines = "\n".join(f"- {c[:200]}" for c in memory_chunks)
        parts.append(f"[MEMORY CONTEXT]\n{mem_lines}\n[END MEMORY]")

    if web_context:
        parts.append(f"[WEB CONTEXT]\n{web_context}\n[END WEB CONTEXT]")

    parts.append(user_input)
    return "\n\n".join(parts)


if __name__ == "__main__":
    result = build_prompt(
        user_input="What do you know about me?",
        memory_chunks=["User's name is Alex.", "User prefers concise answers."],
        web_context=None,
    )
    logger.info(f"Built prompt:\n{result}")
    assert "[MEMORY CONTEXT]" in result
    assert "What do you know about me?" in result
    logger.success("response_builder.py smoke test passed")
