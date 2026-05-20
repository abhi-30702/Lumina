from __future__ import annotations
import threading
import time
import numpy as np
from loguru import logger
import pyaudio
from fuzzywuzzy import fuzz
from voice.transcriber import Transcriber
from config import config

CHUNK = 1024
RATE = 16000
CHANNELS = 1
FORMAT = pyaudio.paInt16
CHUNK_SECONDS = 2


class Listener:
    def __init__(
        self,
        on_utterance: callable,
        on_wake: callable | None = None,
    ) -> None:
        self._on_utterance = on_utterance
        self._on_wake = on_wake or (lambda: None)
        self._tiny = Transcriber(model_name="tiny")
        self._base = Transcriber(model_name=config.whisper_model)
        self._running = False
        self._pa = pyaudio.PyAudio()
        logger.info("Listener initialized")

    def start(self) -> None:
        self._running = True
        thread = threading.Thread(target=self._loop, daemon=True)
        thread.start()
        logger.info("Mic listener started")

    def stop(self) -> None:
        self._running = False

    def _record_chunk(self, stream: pyaudio.Stream, seconds: float) -> np.ndarray:
        frames = []
        n = int(RATE / CHUNK * seconds)
        for _ in range(n):
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(np.frombuffer(data, dtype=np.int16))
        return np.concatenate(frames)

    def _is_wake_word(self, audio: np.ndarray) -> bool:
        text = self._tiny.transcribe(audio.copy()).lower()
        score = fuzz.partial_ratio(config.wake_word, text) / 100.0
        if score >= config.wake_word_threshold:
            logger.info(f"Wake word detected (score={score:.2f}, heard='{text}')")
            return True
        return False

    def _capture_utterance(self, stream: pyaudio.Stream) -> np.ndarray:
        frames = []
        silence_chunks = 0
        silence_limit = int(config.silence_timeout / CHUNK_SECONDS) + 1
        max_chunks = int(config.max_record_seconds / CHUNK_SECONDS)
        for _ in range(max_chunks):
            chunk = self._record_chunk(stream, CHUNK_SECONDS)
            frames.append(chunk)
            rms = np.sqrt(np.mean(chunk.astype(np.float32) ** 2))
            if rms < 300:
                silence_chunks += 1
                if silence_chunks >= silence_limit:
                    break
            else:
                silence_chunks = 0
        return np.concatenate(frames)

    def _loop(self) -> None:
        stream = self._pa.open(
            format=FORMAT, channels=CHANNELS, rate=RATE,
            input=True, frames_per_buffer=CHUNK,
        )
        logger.info("Listening for wake word...")
        while self._running:
            try:
                chunk = self._record_chunk(stream, CHUNK_SECONDS)
                if self._is_wake_word(chunk):
                    self._on_wake()
                    audio = self._capture_utterance(stream)
                    text = self._base.transcribe(audio)
                    if text.strip():
                        logger.info(f"Utterance: '{text}'")
                        self._on_utterance(text)
            except Exception as exc:
                logger.error(f"Listener error: {exc}")
                time.sleep(0.5)
        stream.stop_stream()
        stream.close()


if __name__ == "__main__":
    def handle(text: str) -> None:
        logger.success(f"Got utterance: '{text}'")

    def on_wake() -> None:
        logger.info("Wake word detected!")

    l = Listener(on_utterance=handle, on_wake=on_wake)
    l.start()
    logger.info("Say 'Hey Lumina' ... (running 30s)")
    time.sleep(30)
    l.stop()
    logger.success("listener.py smoke test passed")
