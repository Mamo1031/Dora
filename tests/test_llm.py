"""Test script for local LLM functionality."""

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
        pytest.fail(f"Failed to generate response: {e}")
