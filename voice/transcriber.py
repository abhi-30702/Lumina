from __future__ import annotations
import re
import numpy as np
from loguru import logger
import whisper
from config import config


class Transcriber:
    def __init__(self, model_name: str | None = None) -> None:
        name = model_name or config.whisper_model
        logger.info(f"Loading Whisper model '{name}' ...")
        self._model = whisper.load_model(name)
        logger.success(f"Whisper '{name}' ready")

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32) / 32768.0
        result = self._model.transcribe(
            audio,
            language=config.whisper_language,
            fp16=False,
        )
        text = result.get("text", "").strip()
        text = re.sub(r"[^\w\s'?!.,]", "", text)
        return text

    def transcribe_file(self, path: str) -> str:
        audio = whisper.load_audio(path)
        return self.transcribe(audio)


if __name__ == "__main__":
    import sounddevice as sd
    import soundfile as sf
    from pathlib import Path

    t = Transcriber()
    test_wav = Path("test_audio.wav")
    if test_wav.exists():
        text = t.transcribe_file(str(test_wav))
        logger.info(f"Transcribed: '{text}'")
    else:
        logger.info("Recording 3 seconds of audio for transcription test...")
        audio = sd.rec(int(3 * 16000), samplerate=16000, channels=1, dtype="int16")
        sd.wait()
        text = t.transcribe(audio.flatten())
        logger.info(f"Transcribed: '{text}'")
    logger.success("transcriber.py smoke test passed")
