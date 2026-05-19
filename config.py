from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()


def _int_env(key: str, default: int) -> int:
    val = os.getenv(key, str(default))
    try:
        return int(val)
    except ValueError:
        raise ValueError(f"Config error: {key}={val!r} is not a valid integer")


def _float_env(key: str, default: float) -> float:
    val = os.getenv(key, str(default))
    try:
        return float(val)
    except ValueError:
        raise ValueError(f"Config error: {key}={val!r} is not a valid float")


_VALID_LOG_LEVELS = {"TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"}


@dataclass(frozen=True)
class LuminaConfig:
    # LLM
    ollama_host: str = field(default_factory=lambda: os.getenv("OLLAMA_HOST", "http://localhost:11434"))
    ollama_model: str = field(default_factory=lambda: os.getenv("OLLAMA_MODEL", "lumina-model"))
    ollama_num_gpu: int = field(default_factory=lambda: _int_env("OLLAMA_NUM_GPU", 16))
    ollama_num_thread: int = field(default_factory=lambda: _int_env("OLLAMA_NUM_THREAD", 8))
    llm_max_tokens: int = field(default_factory=lambda: _int_env("LLM_MAX_TOKENS", 512))
    llm_temperature: float = field(default_factory=lambda: _float_env("LLM_TEMPERATURE", 0.7))
    llm_context_window: int = 4096

    # Voice
    whisper_model: str = field(default_factory=lambda: os.getenv("WHISPER_MODEL", "base"))
    whisper_language: str = "en"
    wake_word: str = field(default_factory=lambda: os.getenv("WAKE_WORD", "hey lumina"))
    wake_word_threshold: float = field(default_factory=lambda: _float_env("WAKE_WORD_THRESHOLD", 0.85))
    silence_timeout: float = field(default_factory=lambda: _float_env("SILENCE_TIMEOUT", 1.5))
    max_record_seconds: int = 30
    tts_model: str = field(default_factory=lambda: os.getenv("TTS_MODEL", "tts_models/en/ljspeech/tacotron2-DDC"))

    # Memory
    chroma_path: str = field(default_factory=lambda: os.getenv("CHROMA_PATH", "./memory"))
    memory_top_k: int = field(default_factory=lambda: _int_env("MEMORY_TOP_K", 5))
    conversation_history_turns: int = 10

    # Notes
    notes_path: str = field(default_factory=lambda: os.getenv("NOTES_PATH", "./notes"))

    # UI
    window_width: int = 900
    window_height: int = 600
    always_on_top: bool = field(default_factory=lambda: os.getenv("ALWAYS_ON_TOP", "false").lower() == "true")
    opacity: float = field(default_factory=lambda: _float_env("WINDOW_OPACITY", 0.95))

    # Logging
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    log_path: str = field(default_factory=lambda: os.getenv("LOG_PATH", "./logs"))

    def __post_init__(self) -> None:
        if self.log_level.upper() not in _VALID_LOG_LEVELS:
            raise ValueError(f"Config error: LOG_LEVEL={self.log_level!r} must be one of {sorted(_VALID_LOG_LEVELS)}")

    def resolved_chroma_path(self) -> Path:
        return Path(self.chroma_path).resolve()

    def resolved_notes_path(self) -> Path:
        return Path(self.notes_path).resolve()

    def resolved_log_path(self) -> Path:
        return Path(self.log_path).resolve()


config = LuminaConfig()

if __name__ == "__main__":
    from loguru import logger
    logger.info(f"ollama_model={config.ollama_model}")
    logger.info(f"ollama_num_gpu={config.ollama_num_gpu}")
    logger.info(f"ollama_num_thread={config.ollama_num_thread}")
    logger.info(f"whisper_model={config.whisper_model}")
    logger.info(f"chroma_path={config.resolved_chroma_path()}")
    logger.success("Config loaded OK")
