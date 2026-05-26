from __future__ import annotations
import hashlib
import sys
import threading
from datetime import datetime
from loguru import logger
from PyQt6.QtWidgets import QApplication

from config import config
from core.memory import MemoryManager
from core.brain import Brain
from core.router import classify, SEARCH_ROUTE, NOTE_SAVE_ROUTE, NOTE_READ_ROUTE, NOTE_DELETE_ROUTE, MEMORY_SAVE_ROUTE, MEMORY_FORGET_ROUTE
from core.response_builder import build_prompt
from commands.registry import load_all, match_command
from tools.search import search, format_for_prompt, extract_query
from tools.notes import save_note, read_note, delete_note
from voice.speaker import Speaker
from voice.listener import Listener
from ui.hud import HUDWindow, CYAN, GREEN, AMBER, GRAY, BLUE


def _setup_logging() -> None:
    log_dir = config.resolved_log_path()
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.add(
        str(log_dir / "lumina_{time}.log"),
        level=config.log_level,
        rotation="10 MB",
        retention="7 days",
    )


class LuminaApp:
    def __init__(self, hud: HUDWindow) -> None:
        self._hud = hud
        self._memory = MemoryManager()
        self._brain = Brain()
        self._speaker = Speaker(on_speaking_change=self._on_speaking_change)
        self._session_id = f"sess_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self._standby = False
        self._dispatch_lock = threading.Lock()
        load_all()
        logger.info("LuminaApp initialized")

    def speak(self, text: str) -> None:
        self._speaker.speak(text)

    def _on_speaking_change(self, speaking: bool) -> None:
        self._hud.set_speaking(speaking)
        if speaking:
            self._hud.set_status("SPEAKING", GREEN)
        else:
            self._hud.set_status("LISTENING", CYAN)

    def handle_input(self, text: str) -> None:
        with self._dispatch_lock:
            if self._standby:
                if text.lower().strip().startswith("wake up"):
                    self._standby = False
                    self._hud.set_status("LISTENING", CYAN)
                    self._respond("I'm back. How can I help?")
                return

            self._hud.show_message("user", text)
            self._hud.set_status("THINKING...", AMBER)

            handler, arg = match_command(text)
            if handler:
                result = handler(arg)
                if result == "__STOP_TTS__":
                    self._speaker.stop()
                    self._hud.set_status("LISTENING", CYAN)
                    return
                if result == "__STANDBY__":
                    self._standby = True
                    self._hud.set_status("STANDBY", GRAY)
                    self._respond("Going to standby. Say 'wake up' when you need me.")
                    return
                if result == "__WAKE__":
                    self._standby = False
                    self._hud.set_status("LISTENING", CYAN)
                    self._respond("I'm back.")
                    return
                if result == "__CLEAR_MEMORY__":
                    self._memory.clear_all()
                    self._respond("Memory cleared.")
                    return
                self._respond(result)
                return

            route, arg = classify(text)

            if route == MEMORY_SAVE_ROUTE:
                tag = hashlib.sha1(arg.encode("utf-8")).hexdigest()[:12]
                self._memory.save_fact(arg, tag=tag)
                self._respond(f"Got it. I'll remember that {arg}.")
                return

            if route == MEMORY_FORGET_ROUTE:
                self._memory.forget_fact(arg)
                self._respond(f"Forgotten: {arg}.")
                return

            if route == NOTE_SAVE_ROUTE:
                result = save_note(arg)
                self._respond(result)
                return

            if route == NOTE_READ_ROUTE:
                content = read_note(arg if arg else None)
                self._respond(content[:500])
                return

            if route == NOTE_DELETE_ROUTE:
                result = delete_note(arg)
                self._respond(result)
                return

            web_context: str | None = None
            if route == SEARCH_ROUTE:
                self._hud.set_status("SEARCHING...", BLUE)
                query = extract_query(text)
                results = search(query)
                web_context = format_for_prompt(results) if results else None
                if not results:
                    self._hud.show_message("lumina", "Search unavailable, answering from memory.")

            self._hud.set_status("THINKING...", AMBER)
            context = self._memory.retrieve_context(text)
            prompt = build_prompt(text, context, web_context)
            # Context is empty: build_prompt already injected memory chunks into `prompt`.
            reply = self._brain.chat(prompt, context=[])
            self._memory.save_turn(text, reply, self._session_id)
            self._respond(reply)

    def _respond(self, text: str) -> None:
        self._hud.show_message("lumina", text)
        self._speaker.speak(text)


def main() -> None:
    _setup_logging()
    logger.info("Lumina booting...")
    try:
        app = QApplication(sys.argv)

        hud = HUDWindow()
        lumina = LuminaApp(hud)
        hud.set_text_handler(lumina.handle_input)

        hud.show()
        hud.set_status("LISTENING", CYAN)
        hud.show_message("lumina", "Lumina online. Ready when you are.")

        def on_wake() -> None:
            hud.set_status("ACTIVE", CYAN)

        listener = Listener(
            on_utterance=lumina.handle_input,
            on_wake=on_wake,
        )
        listener.start()
        app.aboutToQuit.connect(listener.stop)

        lumina.speak("Lumina online. Ready when you are.")

        sys.exit(app.exec())
    except Exception:
        logger.exception("Lumina boot failure")
        raise


if __name__ == "__main__":
    main()
