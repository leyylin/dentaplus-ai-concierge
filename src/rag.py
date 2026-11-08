import asyncio
import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb
from loguru import logger

from src.config import settings


@dataclass(slots=True)
class RAGContextResult:
    context: str
    fallback_mode: bool = False

    def __str__(self) -> str:
        return self.context

    def __eq__(self, other: object) -> bool:
        if isinstance(other, str):
            return self.context == other
        if isinstance(other, RAGContextResult):
            return self.context == other.context and self.fallback_mode == other.fallback_mode
        return False


class KnowledgeBaseRAG:
    def __init__(self, kb_path: str | Path = "data/knowledge_base.md") -> None:
        self.kb_path = Path(kb_path)
        self.client = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIR)
        self.collection = self.client.get_or_create_collection(name="dental_kb")

    def _split_markdown(self, text: str) -> list[dict[str, Any]]:
        raw_sections = text.split("## ")
        if not raw_sections:
            return []

        chunks: list[dict[str, Any]] = []

        # Первую часть (до первого ##) пропускаем, если она не содержит полезного контента кроме заголовка H1
        first_part = raw_sections[0].strip()
        first_part_useful = [
            line
            for line in first_part.splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]

        sections = raw_sections[1:]
        if first_part_useful:
            sections = [raw_sections[0]] + sections

        for part in sections:
            clean_part = part.strip()
            if not clean_part:
                continue

            lines = clean_part.splitlines()
            title = lines[0].strip()

            section_text = f"## {clean_part}"

            if len(section_text) <= 700:
                chunk_id = hashlib.sha256(section_text.encode("utf-8")).hexdigest()
                chunks.append(
                    {
                        "id": chunk_id,
                        "text": section_text,
                        "metadata": {"title": title},
                    }
                )
            else:
                body_lines = [l.strip() for l in lines[1:] if l.strip()]
                if not body_lines:
                    chunk_id = hashlib.sha256(section_text.encode("utf-8")).hexdigest()
                    chunks.append(
                        {
                            "id": chunk_id,
                            "text": section_text,
                            "metadata": {"title": title},
                        }
                    )
                    continue

                intro = ""
                items = body_lines
                if len(body_lines[0]) < 200 and not body_lines[0].startswith(
                    ("-", "*", "1.", "2.")
                ):
                    intro = body_lines[0]
                    items = body_lines[1:]

                batch: list[str] = []
                current_len = 0
                for item in items:
                    if batch and (len(batch) >= 4 or current_len + len(item) > 700):
                        subpart_text = (
                            f"{intro}\n" + "\n".join(batch)
                            if intro
                            else "\n".join(batch)
                        )
                        chunk_text = f"## {title}\n{subpart_text.strip()}"
                        chunk_id = hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()
                        chunks.append(
                            {
                                "id": chunk_id,
                                "text": chunk_text,
                                "metadata": {"title": title},
                            }
                        )
                        batch = []
                        current_len = 0

                    batch.append(item)
                    current_len += len(item)

                if batch:
                    subpart_text = (
                        f"{intro}\n" + "\n".join(batch) if intro else "\n".join(batch)
                    )
                    chunk_text = f"## {title}\n{subpart_text.strip()}"
                    chunk_id = hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()
                    chunks.append(
                        {
                            "id": chunk_id,
                            "text": chunk_text,
                            "metadata": {"title": title},
                        }
                    )

        return chunks

    def _index_sync(self) -> int:
        if not self.kb_path.exists():
            logger.warning("Knowledge base file not found: {}", self.kb_path)
            return 0

        content = self.kb_path.read_text(encoding="utf-8")
        chunks = self._split_markdown(content)

        if not chunks:
            logger.warning("No chunks found to index in knowledge base.")
            return 0

        try:
            self.client.delete_collection("dental_kb")
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(name="dental_kb")

        ids = [chunk["id"] for chunk in chunks]
        documents = [chunk["text"] for chunk in chunks]
        metadatas = [chunk["metadata"] for chunk in chunks]

        self.collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
        return len(chunks)

    def _search_sync(
        self,
        query: str,
        top_k: int,
        distance_threshold: float,
    ) -> tuple[list[str], bool]:
        """Синхронный поиск чанков в ChromaDB с проверкой порога расстояния и обработкой ошибок."""
        try:
            if self.collection is None:
                logger.warning("ChromaDB collection is None, activating fallback_mode")
                return [], True

            # Проверка на пустую коллекцию
            if hasattr(self.collection, "count") and self.collection.count() == 0:
                logger.warning("ChromaDB collection is empty, activating fallback_mode")
                return [], True

            results = self.collection.query(
                query_texts=[query],
                n_results=top_k,
                include=["documents", "distances"],
            )

            documents_batch = results.get("documents")
            distances_batch = results.get("distances")

            if not documents_batch or not documents_batch[0]:
                logger.info("No documents returned from ChromaDB query, fallback_mode=True")
                return [], True

            documents = documents_batch[0]
            distances = distances_batch[0] if distances_batch and distances_batch[0] else []

            # Проверка метрики схожести (distance threshold)
            if distances:
                best_distance = distances[0]
                logger.debug("Best distance for query '{}': {}", query, best_distance)
                if best_distance > distance_threshold:
                    logger.warning(
                        "Best chunk distance ({:.4f}) exceeds threshold ({:.4f}) for query '{}', fallback_mode=True",
                        best_distance,
                        distance_threshold,
                        query,
                    )
                    return [], True

            return documents, False

        except Exception as exc:
            logger.error("ChromaDB/sqlite search error for query '{}': {}. Switching to fallback_mode", query, exc)
            return [], True

    async def index_knowledge_base(self) -> int:
        count = await asyncio.to_thread(self._index_sync)
        logger.info("Successfully indexed {} chunks into dental_kb", count)
        return count

    async def retrieve_context(
        self,
        query: str,
        top_k: int | None = None,
        distance_threshold: float | None = None,
    ) -> RAGContextResult:
        k = top_k if top_k is not None else settings.TOP_K_RESULTS
        threshold = (
            distance_threshold
            if distance_threshold is not None
            else getattr(settings, "RAG_DISTANCE_THRESHOLD", 1.2)
        )
        start_time = time.perf_counter()

        try:
            raw_result = await asyncio.to_thread(
                self._search_sync, query, k, threshold
            )
            if isinstance(raw_result, tuple) and len(raw_result) == 2 and isinstance(raw_result[1], bool):
                documents, fallback_mode = raw_result
            elif isinstance(raw_result, list):
                documents, fallback_mode = raw_result, False
            else:
                documents, fallback_mode = [], True
        except Exception as exc:
            logger.error("Error retrieving context for query '{}': {}. Switching to fallback_mode", query, exc)
            documents, fallback_mode = [], True
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        logger.debug("Retrieved chunks for query '{}': {}", query, documents)
        logger.info(
            "Retrieved {} chunks (fallback_mode={}) for query '{}' in {:.2f} ms",
            len(documents),
            fallback_mode,
            query,
            elapsed_ms,
        )
        context_str = "\n\n".join(documents) if documents else ""
        return RAGContextResult(context=context_str, fallback_mode=fallback_mode)
