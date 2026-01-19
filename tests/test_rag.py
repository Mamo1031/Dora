"""Tests for RAG chain module."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from dora.knowledge_base import KnowledgeBase
from dora.rag import RAGChain


@pytest.fixture
def mock_knowledge_base() -> MagicMock:
    """Create a mock knowledge base.

    Returns
    -------
    MagicMock
        Mock knowledge base instance
    """
    kb = MagicMock(spec=KnowledgeBase)
    retriever = MagicMock()
    kb.get_retriever.return_value = retriever
    return kb


@pytest.fixture
def mock_llm() -> MagicMock:
    """Create a mock LLM.

    Returns
    -------
    MagicMock
        Mock LLM instance
    """
    return MagicMock()


def test_rag_chain_initialization(
    mock_knowledge_base: MagicMock,
    mock_llm: MagicMock,
) -> None:
    """Test that RAGChain can be initialized."""
    rag_chain = RAGChain(
        knowledge_base=mock_knowledge_base,
        llm=mock_llm,
        k=4,
    )

    assert rag_chain.knowledge_base == mock_knowledge_base
    assert rag_chain.llm == mock_llm
    assert rag_chain.k == 4
    assert rag_chain.prompt is not None
    assert rag_chain.retriever is not None


def test_rag_chain_invoke(mock_knowledge_base: MagicMock, mock_llm: MagicMock) -> None:
    """Test invoking RAG chain."""
    # Mock retriever
    mock_retriever = MagicMock()
    mock_doc = MagicMock()
    mock_doc.page_content = "Test document content"
    mock_retriever.invoke.return_value = [mock_doc]
    mock_knowledge_base.get_retriever.return_value = mock_retriever

    # Mock LLM
    mock_llm.invoke.return_value = "This is a test answer"

    rag_chain = RAGChain(
        knowledge_base=mock_knowledge_base,
        llm=mock_llm,
    )

    result = rag_chain.invoke("test query")
    assert "result" in result
    assert "source_documents" in result
    assert result["result"] == "This is a test answer"


def test_rag_chain_get_answer(
    mock_knowledge_base: MagicMock,
    mock_llm: MagicMock,
) -> None:
    """Test getting answer from RAG chain."""
    # Mock retriever
    mock_retriever = MagicMock()
    mock_doc = MagicMock()
    mock_doc.page_content = "Test document content"
    mock_retriever.invoke.return_value = [mock_doc]
    mock_knowledge_base.get_retriever.return_value = mock_retriever

    # Mock LLM
    mock_llm.invoke.return_value = "This is a test answer"

    rag_chain = RAGChain(
        knowledge_base=mock_knowledge_base,
        llm=mock_llm,
    )

    answer = rag_chain.get_answer("test query")
    assert answer == "This is a test answer"


def test_rag_chain_with_real_knowledge_base() -> None:
    """Test RAG chain with a real knowledge base (integration test)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        kb = KnowledgeBase(kb_directory=Path(tmpdir) / "kb")

        # Create a document
        doc = Document(
            page_content="Python is a programming language.",
            metadata={"source": "test.pdf"},
        )

        # Add document to knowledge base
        kb.vectorstore.add_documents([doc])

        # Create mock LLM
        with patch("dora.rag.OllamaLLM") as mock_ollama:
            mock_llm = MagicMock()
            mock_ollama.return_value = mock_llm

            rag_chain = RAGChain(knowledge_base=kb, llm=mock_llm, k=1)

            # Mock the LLM's invoke method
            mock_llm.invoke.return_value = "Python is a programming language."

            result = rag_chain.invoke("What is Python?")
            assert "result" in result
            assert "source_documents" in result
            assert result["result"] == "Python is a programming language."
