"""Test script for local LLM functionality."""

from unittest.mock import MagicMock, patch

import pytest

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


@patch("dora.llm.Ollama")
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


@patch("dora.llm.Ollama")
def test_llm_invoke_with_mock(mock_ollama: MagicMock) -> None:
    """Test that LocalLLM invoke works with mocked Ollama."""
    mock_llm_instance = MagicMock()
    mock_llm_instance.invoke.return_value = "Mocked response"
    mock_ollama.return_value = mock_llm_instance

    llm = LocalLLM(model_name="llama3.2")
    response = llm.invoke("Test prompt")

    assert response == "Mocked response"
    mock_llm_instance.invoke.assert_called_once_with("Test prompt")


@patch("dora.llm.Ollama")
def test_llm_invoke_runtime_error(mock_ollama: MagicMock) -> None:
    """Test that LocalLLM raises RuntimeError when invoke fails."""
    mock_llm_instance = MagicMock()
    mock_llm_instance.invoke.side_effect = Exception("Invoke failed")
    mock_ollama.return_value = mock_llm_instance

    llm = LocalLLM(model_name="llama3.2")
    with pytest.raises(RuntimeError) as exc_info:
        llm.invoke("Test prompt")
    assert "Failed to generate response" in str(exc_info.value)
