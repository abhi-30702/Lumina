from __future__ import annotations
from collections import deque
from typing import Generator
from loguru import logger
import ollama
from config import config

SYSTEM_PROMPT = (
    "You are Lumina, an advanced AI personal assistant. "
    "You are intelligent, precise, and speak with calm confidence. "
    "You are female. Your responses are concise unless detail is requested. "
    "You have memory of past conversations. "
    "You never say 'As an AI' or 'I cannot'. "
    "If you don't know something, say 'Let me find that for you'."
)


class Brain:
    def __init__(self) -> None:
        self._history: deque[dict[str, str]] = deque(maxlen=config.conversation_history_turns * 2)
        self._client = ollama.Client(host=config.ollama_host)
        logger.info(f"Brain ready — model={config.ollama_model} num_gpu={config.ollama_num_gpu}")

    def _build_messages(self, user_input: str, context: list[str]) -> list[dict[str, str]]:
        system = SYSTEM_PROMPT
        if context:
            memory_block = "\n".join(f"- {c}" for c in context)
            system = f"{system}\n\n[MEMORY CONTEXT]\n{memory_block}\n[END MEMORY]"
        messages = [{"role": "system", "content": system}]
        messages.extend(self._history)
        messages.append({"role": "user", "content": user_input})
        return messages

    def _options(self) -> dict:
        return {
            "num_gpu": config.ollama_num_gpu,
            "num_thread": config.ollama_num_thread,
            "temperature": config.llm_temperature,
            "num_predict": config.llm_max_tokens,
            "num_ctx": config.llm_context_window,
        }

    def chat(self, user_input: str, context: list[str] | None = None) -> str:
        messages = self._build_messages(user_input, context or [])
        try:
            resp = self._client.chat(
                model=config.ollama_model,
                messages=messages,
                options=self._options(),
            )
            reply = resp["message"]["content"]
        except Exception as exc:
            logger.error(f"Ollama error: {exc}")
            reply = "I'm having trouble thinking right now. Please try again in a moment."
        self._history.append({"role": "user", "content": user_input})
        self._history.append({"role": "assistant", "content": reply})
        return reply

    def stream_chat(self, user_input: str, context: list[str] | None = None) -> Generator[str, None, None]:
        messages = self._build_messages(user_input, context or [])
        full_reply = "I'm having trouble thinking right now. Please try again in a moment."
        try:
            accumulated = ""
            for chunk in self._client.chat(
                model=config.ollama_model,
                messages=messages,
                options=self._options(),
                stream=True,
            ):
                token = chunk["message"]["content"]
                accumulated += token
                yield token
            full_reply = accumulated
        except Exception as exc:
            logger.error(f"Ollama stream error: {exc}")
            yield full_reply
        finally:
            self._history.append({"role": "user", "content": user_input})
            self._history.append({"role": "assistant", "content": full_reply})

    def reset_history(self) -> None:
        self._history.clear()
        logger.info("Conversation history cleared")


if __name__ == "__main__":
    brain = Brain()
    response = brain.chat("Say hello in exactly five words.")
    logger.info(f"Response: {response}")
    logger.success("brain.py smoke test passed")
