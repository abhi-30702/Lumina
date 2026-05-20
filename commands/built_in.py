from __future__ import annotations
from datetime import datetime
from loguru import logger
from commands.registry import lumina_command
from config import config


@lumina_command(trigger=["what time is it", "what's the time"], description="Returns current time")
def get_time(args: str) -> str:
    return f"It is {datetime.now().strftime('%I:%M %p')}."


@lumina_command(trigger=["what's today's date", "what is today's date", "what day is it"], description="Returns current date")
def get_date(args: str) -> str:
    return f"Today is {datetime.now().strftime('%A, %B %d, %Y')}."


@lumina_command(trigger=["help", "what can you do"], description="Lists available commands")
def get_help(args: str) -> str:
    from commands.registry import list_commands
    cmds = "\n".join(f"  • {c}" for c in list_commands())
    return f"Here is what I can do:\n{cmds}"


@lumina_command(trigger=["clear memory", "forget everything"], description="Wipes conversation memory")
def clear_memory(args: str) -> str:
    from core.memory import MemoryManager
    MemoryManager().clear_all()
    return "Memory cleared."


@lumina_command(trigger=["stop", "cancel"], description="Stops current TTS")
def stop_speaking(args: str) -> str:
    return "__STOP_TTS__"


@lumina_command(trigger=["goodbye", "sleep"], description="Puts Lumina in standby")
def go_standby(args: str) -> str:
    return "__STANDBY__"


@lumina_command(trigger=["wake up"], description="Exits standby mode")
def wake_up(args: str) -> str:
    return "__WAKE__"
