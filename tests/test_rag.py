"""
Тесты для RAG-модуля: разбивка markdown на чанки.
"""
import pytest

from src.rag import KnowledgeBaseRAG


def make_rag() -> KnowledgeBaseRAG:
    """Создаёт экземпляр RAG без реальной ChromaDB (тестируем только _split_markdown)."""
    # Monkey-patch: заменяем chromadb.PersistentClient чтобы не создавать файлы на диске
    import unittest.mock as mock
    with mock.patch("chromadb.PersistentClient"):
        return KnowledgeBaseRAG(kb_path="data/knowledge_base.md")


def test_split_markdown_returns_chunks() -> None:
    """_split_markdown должен вернуть непустой список чанков для валидного markdown."""
    rag = make_rag()
    md = """# Заголовок документа

## Раздел 1
Содержимое раздела один.

## Раздел 2
Содержимое раздела два.
"""
    chunks = rag._split_markdown(md)
    assert len(chunks) == 2


def test_split_markdown_skips_top_level_heading() -> None:
    """Заголовок первого уровня (# ...) должен быть пропущен."""
    rag = make_rag()
    md = """# Только заголовок первого уровня
"""
    chunks = rag._split_markdown(md)
    assert chunks == []


def test_split_markdown_chunk_has_required_keys() -> None:
    """Каждый чанк должен содержать ключи id, text, metadata."""
    rag = make_rag()
    md = "## Тестовый раздел\nКакое-то содержимое.\n"
    chunks = rag._split_markdown(md)
    assert len(chunks) == 1
    chunk = chunks[0]
    assert "id" in chunk
    assert "text" in chunk
    assert "metadata" in chunk
    assert chunk["metadata"]["title"] == "Тестовый раздел"


def test_split_markdown_large_section_is_subchunked() -> None:
    """Секция длиннее RAG_MAX_CHUNK_SIZE должна быть разбита на несколько подчанков."""
    rag = make_rag()
    # Генерируем секцию явно длиннее лимита (1200 символов)
    big_paragraph = "А" * 700
    md = f"## Большой раздел\n\n{big_paragraph}\n\n{big_paragraph}\n"
    chunks = rag._split_markdown(md)
    assert len(chunks) > 1


def test_split_markdown_chunk_ids_are_unique() -> None:
    """Все chunk id должны быть уникальными."""
    rag = make_rag()
    md = """## Раздел А
Контент А.

## Раздел Б
Контент Б.

## Раздел В
Контент В.
"""
    chunks = rag._split_markdown(md)
    ids = [c["id"] for c in chunks]
    assert len(ids) == len(set(ids))


def test_index_sync_recreates_collection(tmp_path) -> None:
    """_index_sync должен удалять и заново создавать коллекцию перед добавлением чанков."""
    from unittest.mock import MagicMock
    rag = make_rag()
    test_file = tmp_path / "test_kb.md"
    test_file.write_text("## Секция 1\nТекст 1\n", encoding="utf-8")
    rag.kb_path = test_file

    rag.client = MagicMock()
    mock_coll = MagicMock()
    rag.client.get_or_create_collection.return_value = mock_coll

    count = rag._index_sync()

    rag.client.delete_collection.assert_called_once_with("dental_kb")
    rag.client.get_or_create_collection.assert_called_once_with(name="dental_kb")
    mock_coll.upsert.assert_called_once()
    assert count == 1


def test_split_markdown_subchunks_have_title_prefix_and_sha256_id() -> None:
    """Для секций > 700 символов каждый подчанк должен иметь префикс ## {title} и sha256 id."""
    import hashlib
    rag = make_rag()
    lines = [f"- [ART-{100+i}] Услуга номер {i} описание процедуры — {1000*i} руб." for i in range(12)]
    md = "## Прайс-лист и услуги\nВводный текст списка услуг:\n" + "\n".join(lines) + "\n"
    assert len(md) > 700

    chunks = rag._split_markdown(md)
    assert len(chunks) >= 2
    for c in chunks:
        assert c["text"].startswith("## Прайс-лист и услуги\n")
        assert c["id"] == hashlib.sha256(c["text"].encode("utf-8")).hexdigest()
        assert c["metadata"]["title"] == "Прайс-лист и услуги"


@pytest.mark.asyncio
async def test_retrieve_context_debug_logging() -> None:
    """retrieve_context должен вызывать logger.debug с точным списком чанков."""
    from unittest.mock import patch
    rag = make_rag()
    with patch.object(rag, "_search_sync", return_value=["Чанк 1", "Чанк 2"]):
        with patch("src.rag.logger.debug") as mock_debug:
            result = await rag.retrieve_context("вопрос о ценах", top_k=2)
            assert "Чанк 1\n\nЧанк 2" == result
            mock_debug.assert_called_once_with(
                "Retrieved chunks for query '{}': {}",
                "вопрос о ценах",
                ["Чанк 1", "Чанк 2"],
            )


def test_search_sync_distance_threshold_triggers_fallback() -> None:
    """Если дистанция первого чанка превышает пороговое значение, возвращается fallback_mode=True."""
    from unittest.mock import MagicMock
    rag = make_rag()
    rag.collection = MagicMock()
    rag.collection.count.return_value = 5
    rag.collection.query.return_value = {
        "documents": [["Нерелевантный чанк"]],
        "distances": [[1.85]],
    }

    docs, fallback_mode = rag._search_sync("запрос", top_k=1, distance_threshold=1.2)
    assert docs == []
    assert fallback_mode is True


def test_search_sync_empty_collection_triggers_fallback() -> None:
    """Если коллекция пуста, возвращается пустой контекст и fallback_mode=True."""
    from unittest.mock import MagicMock
    rag = make_rag()
    rag.collection = MagicMock()
    rag.collection.count.return_value = 0

    docs, fallback_mode = rag._search_sync("запрос", top_k=1, distance_threshold=1.2)
    assert docs == []
    assert fallback_mode is True


def test_search_sync_db_exception_triggers_fallback() -> None:
    """При исключении ChromaDB/sqlite возвращается пустой контекст и fallback_mode=True."""
    from unittest.mock import MagicMock
    rag = make_rag()
    rag.collection = MagicMock()
    rag.collection.count.side_effect = RuntimeError("Database disk image is malformed")

    docs, fallback_mode = rag._search_sync("запрос", top_k=1, distance_threshold=1.2)
    assert docs == []
    assert fallback_mode is True
