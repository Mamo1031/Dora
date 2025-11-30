"""CLI interface for Dora."""

import sys

from dora.llm import LocalLLM


def run_test_llm() -> None:
    """Test the local LLM with a simple prompt."""
    print("Initializing Local LLM with Llama 3.2 (3B)...")  # noqa: T201
    try:
        llm = LocalLLM(model_name="llama3.2")
        print("✓ LLM initialized successfully\n")  # noqa: T201
    except ConnectionError as e:
        print(f"✗ Error: {e}")  # noqa: T201
        print("\nPlease ensure:")  # noqa: T201
        print("  1. Ollama is installed and running")  # noqa: T201
        print("  2. Llama 3.2 model is pulled: ollama pull llama3.2")  # noqa: T201
        sys.exit(1)

    # Default test prompt
    default_prompt = "Hello! Please introduce yourself in one sentence."
    print(f"Test prompt: {default_prompt}\n")  # noqa: T201
    print("Generating response...\n")  # noqa: T201

    try:
        response = llm.invoke(default_prompt)
        print("Response:")  # noqa: T201
        print("-" * 50)  # noqa: T201
        print(response)  # noqa: T201
        print("-" * 50)  # noqa: T201
    except RuntimeError as e:
        print(f"✗ Error generating response: {e}")  # noqa: T201
        sys.exit(1)
