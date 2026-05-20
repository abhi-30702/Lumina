from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable
from loguru import logger

_REGISTRY: list["Command"] = []


@dataclass
class Command:
    triggers: list[str]
    description: str
    handler: Callable[[str], str]


def lumina_command(trigger: list[str], description: str = ""):
    """Decorator that registers a function as a Lumina command."""
    def decorator(fn: Callable[[str], str]) -> Callable[[str], str]:
        cmd = Command(triggers=trigger, description=description, handler=fn)
        _REGISTRY.append(cmd)
        logger.debug(f"Registered command: {trigger[0]}")
        return fn
    return decorator


def match_command(text: str) -> tuple[Callable[[str], str] | None, str]:
    """Return (handler, arg) if input matches a command trigger, else (None, '')."""
    lower = text.lower().strip()
    for cmd in _REGISTRY:
        for trigger in cmd.triggers:
            if lower.startswith(trigger):
                arg = lower[len(trigger):].strip()
                return cmd.handler, arg
    return None, ""


def list_commands() -> list[str]:
    return [f"{c.triggers[0]}: {c.description}" for c in _REGISTRY]


def load_all() -> None:
    import commands.built_in  # noqa: F401
    try:
        import commands.user_commands  # noqa: F401
    except Exception as exc:
        logger.warning(f"user_commands load error: {exc}")
    logger.info(f"Loaded {len(_REGISTRY)} commands")
