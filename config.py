from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import os
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

@dataclass
class LuminaConfig:
    # LLM
    ollama_host: str = field(default_factory=lambda: os.getenv("OLLAMA_HOST", "http://localhost:11434"))
    ollama_model: str = field(default_factory=lambda: os.getenv("OLLAMA_MODEL", "lumina-model"))
    ollama_num_gpu: int = field(default_factory=lambda: int(os.getenv("OLLAMA_NUM_GPU", "16")))
    ollama_num_thread: int = field(default_factory=lambda: int(os.getenv("OLLAMA_NUM_THREAD", "8")))
    llm_max_tokens: int = field(default_factory=lambda: int(os.getenv("LLM_MAX_TOKENS", "512")))
    llm_temperature: float = field(default_factory=lambda: float(os.getenv("LLM_TEMPERATURE", "0.7")))
    llm_context_window: int = 4096

    # Voice
    whisper_model: str = field(default_factory=lambda: os.getenv("WHISPER_MODEL", "base"))
    whisper_language: str = "en"
    wake_word: str = field(default_factory=lambda: os.getenv("WAKE_WORD", "hey lumina"))
    wake_word_threshold: float = field(default_factory=lambda: float(os.getenv("WAKE_WORD_THRESHOLD", "0.85")))
    silence_timeout: float = field(default_factory=lambda: float(os.getenv("SILENCE_TIMEOUT", "1.5")))
    max_record_seconds: int = 30
    tts_model: str = field(default_factory=lambda: os.getenv("TTS_MODEL", "tts_models/en/ljspeech/tacotron2-DDC"))

    # Memory
    chroma_path: str = field(default_factory=lambda: os.getenv("CHROMA_PATH", "./memory"))
    memory_top_k: int = field(default_factory=lambda: int(os.getenv("MEMORY_TOP_K", "5")))
    conversation_history_turns: int = 10

    # Notes
    notes_path: str = field(default_factory=lambda: os.getenv("NOTES_PATH", "./notes"))

    # UI
    window_width: int = 900
    window_height: int = 600
    always_on_top: bool = field(default_factory=lambda: os.getenv("ALWAYS_ON_TOP", "false").lower() == "true")
    opacity: float = field(default_factory=lambda: float(os.getenv("WINDOW_OPACITY", "0.95")))

    # Logging
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    log_path: str = field(default_factory=lambda: os.getenv("LOG_PATH", "./logs"))

    def resolved_chroma_path(self) -> Path:
        return Path(self.chroma_path).resolve()

    def resolved_notes_path(self) -> Path:
        return Path(self.notes_path).resolve()

    def resolved_log_path(self) -> Path:
        return Path(self.log_path).resolve()


config = LuminaConfig()

if __name__ == "__main__":
    logger.info(f"ollama_model={config.ollama_model}")
    logger.info(f"ollama_num_gpu={config.ollama_num_gpu}")
    logger.info(f"ollama_num_thread={config.ollama_num_thread}")
    logger.info(f"whisper_model={config.whisper_model}")
    logger.info(f"chroma_path={config.resolved_chroma_path()}")
    logger.success("Config loaded OK")
