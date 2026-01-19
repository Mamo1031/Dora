"""Test script for local LLM functionality."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from dora.knowledge_base import KnowledgeBase
from dora.llm import LocalLLM


def test_llm_initialization() -> None:
    """Test that LocalLLM can be initialized."""
    try:
        llm = LocalLLM(model_name="llama3.2")
        assert llm.model_name == "llama3.2"
        assert llm.llm is not None
    except ConnectionError as e:
        pytest.skip(f"Ollama not available: {e}")


def test_llm_initialization_custom_model() -> None:
    """Test that LocalLLM can be initialized with a custom model name."""
    try:
        llm = LocalLLM(model_name="custom-model")
        assert llm.model_name == "custom-model"
        assert llm.llm is not None
    except ConnectionError as e:
        pytest.skip(f"Ollama not available: {e}")


def test_llm_initialization_default_model() -> None:
    """Test that LocalLLM uses default model name when not specified."""
    try:
        llm = LocalLLM()
        assert llm.model_name == "llama3.2"
        assert llm.llm is not None
    except ConnectionError as e:
        pytest.skip(f"Ollama not available: {e}")


@patch("dora.llm.OllamaLLM")
def test_llm_initialization_connection_error(mock_ollama: MagicMock) -> None:
    """Test that LocalLLM raises ConnectionError when Ollama is not available."""
    mock_ollama.side_effect = Exception("Connection refused")
    with pytest.raises(ConnectionError) as exc_info:
        LocalLLM(model_name="llama3.2")
    assert "Failed to connect to Ollama" in str(exc_info.value)
    assert "llama3.2" in str(exc_info.value)


def test_llm_invoke() -> None:
    """Test that LocalLLM can generate a response."""
    try:
        llm = LocalLLM(model_name="llama3.2")
    except ConnectionError as e:
        pytest.skip(f"Ollama not available: {e}")

    default_prompt = "Hello! Please introduce yourself in one sentence."
    try:
        response = llm.invoke(default_prompt)
        assert isinstance(response, str)
        assert len(response) > 0
    except RuntimeError as e:
        # Check if the error is due to connection issues (Ollama not available)
        error_str = str(e).lower()
        if "connection" in error_str or "connection refused" in error_str:
            pytest.skip(f"Ollama not available during invoke: {e}")
        pytest.fail(f"Failed to generate response: {e}")


@patch("dora.llm.OllamaLLM")
def test_llm_invoke_with_mock(mock_ollama: MagicMock) -> None:
    """Test that LocalLLM invoke works with mocked Ollama."""
    mock_llm_instance = MagicMock()
    mock_llm_instance.invoke.return_value = "Mocked response"
    mock_ollama.return_value = mock_llm_instance

    llm = LocalLLM(model_name="llama3.2")
    response = llm.invoke("Test prompt")

    assert response == "Mocked response"
    mock_llm_instance.invoke.assert_called_once_with("Test prompt")


@patch("dora.llm.OllamaLLM")
def test_llm_invoke_runtime_error(mock_ollama: MagicMock) -> None:
    """Test that LocalLLM raises RuntimeError when invoke fails."""
    mock_llm_instance = MagicMock()
    mock_llm_instance.invoke.side_effect = Exception("Invoke failed")
    mock_ollama.return_value = mock_llm_instance

    llm = LocalLLM(model_name="llama3.2")
    with pytest.raises(RuntimeError) as exc_info:
        llm.invoke("Test prompt")
    assert "Failed to generate response" in str(exc_info.value)


@patch("dora.llm.OllamaLLM")
def test_llm_initialization_with_rag(mock_ollama: MagicMock) -> None:
    """Test that LocalLLM can be initialized with RAG enabled."""
    mock_llm_instance = MagicMock()
    mock_ollama.return_value = mock_llm_instance

    with tempfile.TemporaryDirectory() as tmpdir:
        kb = KnowledgeBase(kb_directory=Path(tmpdir) / "kb")
        llm = LocalLLM(model_name="llama3.2", use_rag=True, knowledge_base=kb)

        assert llm.use_rag is True
        assert llm.knowledge_base == kb
        assert llm.rag_chain is not None


@patch("dora.llm.OllamaLLM")
def test_llm_initialization_rag_without_kb(mock_ollama: MagicMock) -> None:
    """Test that LocalLLM raises ValueError when RAG is enabled without knowledge base."""
    mock_llm_instance = MagicMock()
    mock_ollama.return_value = mock_llm_instance

    with pytest.raises(ValueError, match="knowledge_base must be provided"):
        LocalLLM(model_name="llama3.2", use_rag=True, knowledge_base=None)


@patch("dora.llm.OllamaLLM")
@patch("dora.llm.RAGChain")
def test_llm_invoke_with_rag(mock_rag_chain_class: MagicMock, mock_ollama: MagicMock) -> None:
    """Test that LocalLLM uses RAG when enabled."""
    mock_llm_instance = MagicMock()
    mock_ollama.return_value = mock_llm_instance

    mock_rag_chain = MagicMock()
    mock_rag_chain.get_answer.return_value = "RAG answer"
    mock_rag_chain_class.return_value = mock_rag_chain

    with tempfile.TemporaryDirectory() as tmpdir:
        kb = KnowledgeBase(kb_directory=Path(tmpdir) / "kb")
        llm = LocalLLM(model_name="llama3.2", use_rag=True, knowledge_base=kb)

        response = llm.invoke("Test query")
        assert response == "RAG answer"
        mock_rag_chain.get_answer.assert_called_once_with("Test query")


@patch("dora.llm.OllamaLLM")
@patch("dora.llm.RAGChain")
def test_llm_invoke_with_sources(mock_rag_chain_class: MagicMock, mock_ollama: MagicMock) -> None:
    """Test that LocalLLM can return response with sources."""
    mock_llm_instance = MagicMock()
    mock_ollama.return_value = mock_llm_instance

    mock_rag_chain = MagicMock()
    mock_rag_chain.invoke.return_value = {
        "result": "RAG answer",
        "source_documents": [],
    }
    mock_rag_chain_class.return_value = mock_rag_chain

    with tempfile.TemporaryDirectory() as tmpdir:
        kb = KnowledgeBase(kb_directory=Path(tmpdir) / "kb")
        llm = LocalLLM(model_name="llama3.2", use_rag=True, knowledge_base=kb)

        result = llm.invoke_with_sources("Test query")
        assert "result" in result
        assert "source_documents" in result
        assert result["result"] == "RAG answer"


@patch("dora.llm.OllamaLLM")
def test_llm_invoke_with_sources_without_rag(mock_ollama: MagicMock) -> None:
    """Test that invoke_with_sources raises ValueError when RAG is not enabled."""
    mock_llm_instance = MagicMock()
    mock_ollama.return_value = mock_llm_instance

    llm = LocalLLM(model_name="llama3.2", use_rag=False)
    with pytest.raises(ValueError, match="RAG must be enabled"):
        llm.invoke_with_sources("Test query")


@patch("dora.llm.OllamaLLM")
@patch("dora.llm.RAGChain")
def test_llm_invoke_with_sources_runtime_error(
    mock_rag_chain_class: MagicMock,
    mock_ollama: MagicMock,
) -> None:
    """Test that invoke_with_sources raises RuntimeError on failure."""
    mock_llm_instance = MagicMock()
    mock_ollama.return_value = mock_llm_instance

    mock_rag_chain = MagicMock()
    mock_rag_chain.invoke.side_effect = Exception("RAG invoke failed")
    mock_rag_chain_class.return_value = mock_rag_chain

    with tempfile.TemporaryDirectory() as tmpdir:
        kb = KnowledgeBase(kb_directory=Path(tmpdir) / "kb")
        llm = LocalLLM(model_name="llama3.2", use_rag=True, knowledge_base=kb)

        with pytest.raises(RuntimeError, match="Failed to generate response"):
            llm.invoke_with_sources("Test query")


@patch("dora.llm.OllamaLLM")
def test_llm_invoke_with_performance_without_rag(mock_ollama: MagicMock) -> None:
    """Test invoke_with_performance without RAG."""
    mock_llm_instance = MagicMock()
    mock_llm_instance.invoke.return_value = "Test response"
    mock_ollama.return_value = mock_llm_instance

    llm = LocalLLM(model_name="llama3.2", use_rag=False)
    response, performance = llm.invoke_with_performance("Test prompt")

    assert response == "Test response"
    assert "total_time" in performance
    assert "generation_time" in performance
    assert "retrieval_time" in performance
    assert "ttft" in performance
    assert performance["retrieval_time"] == 0.0
    assert performance["total_time"] > 0
    mock_llm_instance.invoke.assert_called_once_with("Test prompt")


@patch("dora.llm.OllamaLLM")
@patch("dora.llm.RAGChain")
def test_llm_invoke_with_performance_with_rag(
    mock_rag_chain_class: MagicMock,
    mock_ollama: MagicMock,
) -> None:
    """Test invoke_with_performance with RAG enabled."""
    mock_llm_instance = MagicMock()
    mock_ollama.return_value = mock_llm_instance

    mock_rag_chain = MagicMock()
    mock_rag_chain.get_answer_with_performance.return_value = (
        "RAG answer",
        {"generation_time": 1.0, "retrieval_time": 0.5},
    )
    mock_rag_chain_class.return_value = mock_rag_chain

    with tempfile.TemporaryDirectory() as tmpdir:
        kb = KnowledgeBase(kb_directory=Path(tmpdir) / "kb")
        llm = LocalLLM(model_name="llama3.2", use_rag=True, knowledge_base=kb)

        response, performance = llm.invoke_with_performance("Test query")

        assert response == "RAG answer"
        assert "total_time" in performance
        assert "generation_time" in performance
        assert "retrieval_time" in performance
        assert "ttft" in performance
        assert performance["generation_time"] == 1.0
        assert performance["retrieval_time"] == 0.5
        mock_rag_chain.get_answer_with_performance.assert_called_once_with("Test query")


@patch("dora.llm.OllamaLLM")
def test_llm_invoke_with_performance_runtime_error(mock_ollama: MagicMock) -> None:
    """Test invoke_with_performance raises RuntimeError on failure."""
    mock_llm_instance = MagicMock()
    mock_llm_instance.invoke.side_effect = Exception("Invoke failed")
    mock_ollama.return_value = mock_llm_instance

    llm = LocalLLM(model_name="llama3.2", use_rag=False)

    with pytest.raises(RuntimeError, match="Failed to generate response"):
        llm.invoke_with_performance("Test prompt")
