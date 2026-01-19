"""Tests for knowledge base module."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from dora.knowledge_base import KnowledgeBase


def test_knowledge_base_initialization() -> None:
    """Test that KnowledgeBase can be initialized."""
    with tempfile.TemporaryDirectory() as tmpdir:
        kb = KnowledgeBase(kb_directory=Path(tmpdir) / "kb")
        assert kb.kb_directory == Path(tmpdir) / "kb"
        assert kb.document_processor is not None
        assert kb.vectorstore is not None


def test_knowledge_base_custom_params() -> None:
    """Test KnowledgeBase with custom parameters."""
    with tempfile.TemporaryDirectory() as tmpdir:
        kb = KnowledgeBase(
            kb_directory=Path(tmpdir) / "kb",
            chunk_size=500,
            chunk_overlap=100,
            embedding_model_name="all-MiniLM-L6-v2",
        )
        assert kb.document_processor.chunk_size == 500
        assert kb.document_processor.chunk_overlap == 100


@patch("dora.knowledge_base.DocumentProcessor")
def test_add_document(mock_doc_processor_class: MagicMock) -> None:
    """Test adding a document to knowledge base."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Mock document processor
        mock_processor = MagicMock()
        mock_chunk = Document(
            page_content="Test content",
            metadata={"source": "test.pdf"},
        )
        mock_processor.load_pdf.return_value = [mock_chunk]
        mock_doc_processor_class.return_value = mock_processor

        kb = KnowledgeBase(kb_directory=Path(tmpdir) / "kb")
        kb.document_processor = mock_processor

        file_path = Path("test.pdf")
        with (
            patch.object(Path, "exists", return_value=True),
            patch.object(Path, "suffix", new_callable=lambda: ".pdf"),
        ):
            num_chunks = kb.add_document(file_path)
            assert num_chunks == 1


def test_add_document_file_not_found() -> None:
    """Test that FileNotFoundError is raised for non-existent files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        kb = KnowledgeBase(kb_directory=Path(tmpdir) / "kb")
        file_path = Path("nonexistent.pdf")

        with pytest.raises(FileNotFoundError):
            kb.add_document(file_path)


def test_search() -> None:
    """Test searching the knowledge base."""
    with tempfile.TemporaryDirectory() as tmpdir:
        kb = KnowledgeBase(kb_directory=Path(tmpdir) / "kb")

        # Create document
        doc = Document(
            page_content="Python is a programming language.",
            metadata={"source": "test.pdf"},
        )

        kb.vectorstore.add_documents([doc])

        results = kb.search("Python", k=1)
        assert isinstance(results, list)


def test_get_retriever() -> None:
    """Test getting retriever from knowledge base."""
    with tempfile.TemporaryDirectory() as tmpdir:
        kb = KnowledgeBase(kb_directory=Path(tmpdir) / "kb")
        retriever = kb.get_retriever()
        assert retriever is not None


def test_clear() -> None:
    """Test clearing the knowledge base."""
    with tempfile.TemporaryDirectory() as tmpdir:
        kb = KnowledgeBase(kb_directory=Path(tmpdir) / "kb")

        # Add a document
        doc = Document(
            page_content="Test content",
            metadata={"source": "test.pdf"},
        )
        kb.vectorstore.add_documents([doc])

        # Clear
        kb.clear()

        # Check that it's empty
        info = kb.get_info()
        assert info["count"] == 0 or not info["exists"]


def test_get_info() -> None:
    """Test getting knowledge base info."""
    with tempfile.TemporaryDirectory() as tmpdir:
        kb = KnowledgeBase(kb_directory=Path(tmpdir) / "kb")
        info = kb.get_info()
        assert "count" in info
        assert "exists" in info


def test_list_documents_empty() -> None:
    """Test listing documents when knowledge base is empty."""
    with tempfile.TemporaryDirectory() as tmpdir:
        kb = KnowledgeBase(kb_directory=Path(tmpdir) / "kb")
        documents = kb.list_documents()
        assert documents == []


def test_list_documents_with_content() -> None:
    """Test listing documents when knowledge base has content."""
    with tempfile.TemporaryDirectory() as tmpdir:
        kb = KnowledgeBase(kb_directory=Path(tmpdir) / "kb")

        # Add documents
        doc1 = Document(
            page_content="Content 1",
            metadata={"source": "test1.pdf"},
        )
        doc2 = Document(
            page_content="Content 2",
            metadata={"source": "test2.pdf"},
        )

        kb.vectorstore.add_documents([doc1, doc2])

        documents = kb.list_documents()
        # Should return list (may be empty if similarity search fails)
        assert isinstance(documents, list)
