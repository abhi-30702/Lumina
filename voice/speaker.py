from __future__ import annotations
import queue
import tempfile
import threading
from pathlib import Path
from typing import Callable
from loguru import logger
import sounddevice as sd
import soundfile as sf
from TTS.api import TTS
from config import config


class Speaker:
    def __init__(self, on_speaking_change: Callable[[bool], None] | None = None) -> None:
        self._on_speaking_change = on_speaking_change or (lambda _: None)
        self._queue: queue.Queue[str] = queue.Queue()
        self._is_speaking = False
        self._stop_flag = threading.Event()
        logger.info(f"Loading TTS model '{config.tts_model}' ...")
        # CPU-only TTS reserves VRAM for Ollama (16 layers of Mistral) + Whisper
        self._tts = TTS(model_name=config.tts_model, progress_bar=False, gpu=False)
        logger.success("TTS ready")
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def speak(self, text: str) -> None:
        self._queue.put(text)

    def stop(self) -> None:
        self._stop_flag.set()
        sd.stop()
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    @property
    def is_speaking(self) -> bool:
        return self._is_speaking

    def _emit_speaking(self, speaking: bool) -> None:
        try:
            self._on_speaking_change(speaking)
        except Exception as exc:
            logger.error(f"on_speaking_change callback error: {exc}")

    def _worker(self) -> None:
        while True:
            text = self._queue.get()
            if not text:
                continue
            self._stop_flag.clear()
            self._is_speaking = True
            self._emit_speaking(True)
            tmp_path: str | None = None
            try:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    tmp_path = f.name
                self._tts.tts_to_file(text=text, file_path=tmp_path)
                if not self._stop_flag.is_set():
                    data, sr = sf.read(tmp_path)
                    sd.play(data, sr)
                    sd.wait()
            except Exception as exc:
                logger.error(f"TTS error: {exc}")
            finally:
                if tmp_path:
                    Path(tmp_path).unlink(missing_ok=True)
                self._is_speaking = False
                self._emit_speaking(False)


if __name__ == "__main__":
    import time

    def on_change(speaking: bool) -> None:
        logger.info(f"Speaking: {speaking}")

    s = Speaker(on_speaking_change=on_change)
    s.speak("Hello. I am Lumina, your AI assistant. How can I help you today?")
    time.sleep(10)
    logger.success("speaker.py smoke test passed")
