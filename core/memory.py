from __future__ import annotations
import uuid
from datetime import datetime
from loguru import logger
import chromadb
from sentence_transformers import SentenceTransformer
from config import config


class MemoryManager:
    def __init__(self) -> None:
        path = str(config.resolved_chroma_path())
        self._client = chromadb.PersistentClient(path=path)
        self._embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        self._conversations = self._client.get_or_create_collection("conversations")
        self._facts = self._client.get_or_create_collection("facts")
        logger.info(f"MemoryManager ready at {path}")

    def _embed(self, text: str) -> list[float]:
        return self._embed_model.encode(text).tolist()

    def save_turn(self, user_msg: str, lumina_msg: str, session_id: str = "default") -> None:
        try:
            doc = f"User: {user_msg}\nLumina: {lumina_msg}"
            ts = datetime.now().isoformat()
            self._conversations.add(
                ids=[f"conv_{ts}_{uuid.uuid4().hex[:6]}"],
                documents=[doc],
                embeddings=[self._embed(doc)],
                metadatas=[{
                    "timestamp": ts,
                    "user_msg": user_msg,
                    "lumina_msg": lumina_msg,
                    "session_id": session_id,
                }],
            )
        except Exception as exc:
            logger.error(f"Failed to save turn: {exc}")
            raise

    def save_fact(self, fact: str, tag: str) -> None:
        try:
            ts = datetime.now().isoformat()
            self._facts.upsert(
                ids=[f"fact_{tag}"],
                documents=[fact],
                embeddings=[self._embed(fact)],
                metadatas=[{"tag": tag, "timestamp": ts, "source": "explicit"}],
            )
            logger.info(f"Saved fact tag={tag}")
        except Exception as exc:
            logger.error(f"Failed to save fact tag={tag}: {exc}")
            raise

    def retrieve_context(self, query: str, top_k: int | None = None) -> list[str]:
        k = top_k or config.memory_top_k
        emb = self._embed(query)
        results: list[str] = []

        conv_k = max(1, k - 2)
        fact_k = min(k, 2)

        conv_count = self._conversations.count()
        if conv_count > 0:
            conv = self._conversations.query(query_embeddings=[emb], n_results=min(conv_k, conv_count))
            if conv["documents"]:
                results.extend(conv["documents"][0])

        facts_count = self._facts.count()
        if facts_count > 0:
            facts = self._facts.query(query_embeddings=[emb], n_results=min(fact_k, facts_count))
            if facts["documents"]:
                results.extend(facts["documents"][0])

        return results[:k]

    def forget_fact(self, tag: str) -> None:
        self._facts.delete(ids=[f"fact_{tag}"])
        logger.info(f"Deleted fact tag={tag}")

    def clear_all(self) -> None:
        self._client.delete_collection("conversations")
        self._client.delete_collection("facts")
        self._conversations = self._client.get_or_create_collection("conversations")
        self._facts = self._client.get_or_create_collection("facts")
        logger.warning("All memory cleared")


if __name__ == "__main__":
    mem = MemoryManager()
    mem.save_turn("What is Python?", "Python is a high-level programming language.")
    mem.save_fact("The user's name is Alex.", "name")
    results = mem.retrieve_context("Python programming")
    logger.info(f"Retrieved {len(results)} chunks:")
    for r in results:
        logger.info(f"  - {r[:80]}")
    mem.forget_fact("name")
    logger.success("memory.py smoke test passed")
