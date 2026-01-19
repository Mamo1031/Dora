"""CLI interface for Dora."""

import sys
from pathlib import Path

from dora.knowledge_base import KnowledgeBase
from dora.llm import LocalLLM

MIN_ARGS_FOR_ADD_DOC = 3


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


def _initialize_knowledge_base() -> tuple[KnowledgeBase | None, bool]:
    """Initialize knowledge base and determine if RAG should be used.

    Returns
    -------
    tuple[KnowledgeBase | None, bool]
        Tuple of (knowledge_base, use_rag)
    """
    try:
        kb = KnowledgeBase()
        kb_info = kb.get_info()
        if kb_info["exists"] and kb_info["count"] > 0:
            print(f"✓ Knowledge base found ({kb_info['count']} chunks)")  # noqa: T201
            print("RAG mode enabled\n")  # noqa: T201
            return kb, True
        print("i No documents in knowledge base. RAG disabled.")  # noqa: T201
        print("  Use 'dora add-doc <file>' to add documents\n")  # noqa: T201
        return kb, False
    except (OSError, ValueError, RuntimeError):
        print("i Could not initialize knowledge base. RAG disabled.\n")  # noqa: T201
        return None, False


def _initialize_llm(use_rag: bool, kb: KnowledgeBase | None) -> LocalLLM:
    """Initialize the LLM with optional RAG support.

    Parameters
    ----------
    use_rag : bool
        Whether to use RAG
    kb : KnowledgeBase | None
        Knowledge base instance

    Returns
    -------
    LocalLLM
        Initialized LLM instance

    Note
    ----
    This function may call sys.exit() if initialization fails.
    """
    try:
        return LocalLLM(model_name="llama3.2", use_rag=use_rag, knowledge_base=kb)
    except ConnectionError as e:
        print(f"✗ Error: {e}")  # noqa: T201
        print("\nPlease ensure:")  # noqa: T201
        print("  1. Ollama is installed and running")  # noqa: T201
        print("  2. Llama 3.2 model is pulled: ollama pull llama3.2")  # noqa: T201
        sys.exit(1)
    except ValueError as e:
        print(f"✗ Error: {e}")  # noqa: T201
        sys.exit(1)


def interactive() -> None:
    """Interactive CLI for chatting with the local LLM with RAG support."""
    print("Initializing Local LLM with Llama 3.2 (3B)...")  # noqa: T201

    kb, use_rag = _initialize_knowledge_base()
    llm = _initialize_llm(use_rag, kb)
    print("✓ LLM initialized successfully\n")  # noqa: T201

    print("Interactive mode started. Type 'exit' or 'quit' to end the session.\n")  # noqa: T201

    while True:
        try:
            prompt = input("You: ").strip()
            if not prompt:
                continue

            if prompt.lower() in {"exit", "quit"}:
                print("\nGoodbye!")  # noqa: T201
                break

            print("Generating response...")  # noqa: T201
            try:
                response = llm.invoke(prompt)
                print(f"Assistant: {response}\n")  # noqa: T201
            except RuntimeError as e:
                print(f"✗ Error generating response: {e}\n")  # noqa: T201

        except KeyboardInterrupt:
            print("\n\nGoodbye!")  # noqa: T201
            break
        except EOFError:
            print("\n\nGoodbye!")  # noqa: T201
            break


def add_document() -> None:
    """Add a document to the knowledge base."""
    if len(sys.argv) < MIN_ARGS_FOR_ADD_DOC:
        print("Usage: dora add-doc <file_path>")  # noqa: T201
        sys.exit(1)

    file_path = Path(sys.argv[2])

    if not file_path.exists():
        print(f"✗ Error: File not found: {file_path}")  # noqa: T201
        sys.exit(1)

    if file_path.suffix.lower() != ".pdf":
        print(f"✗ Error: Only PDF files are supported: {file_path}")  # noqa: T201
        sys.exit(1)

    print(f"Adding document: {file_path}")  # noqa: T201
    print("Processing document...")  # noqa: T201

    try:
        kb = KnowledgeBase()
        num_chunks = kb.add_document(file_path)
        print(f"✓ Document added successfully ({num_chunks} chunks)")  # noqa: T201
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"✗ Error adding document: {e}")  # noqa: T201
        sys.exit(1)


def list_documents() -> None:
    """List documents in the knowledge base."""
    try:
        kb = KnowledgeBase()
        info = kb.get_info()

        if not info["exists"] or info["count"] == 0:
            print("No documents in knowledge base.")  # noqa: T201
            print("Use 'dora add-doc <file>' to add documents.")  # noqa: T201
            return

        print(f"Knowledge base contains {info['count']} chunks.")  # noqa: T201
        print("\nDocuments:")  # noqa: T201

        documents = kb.list_documents()
        if documents:
            for i, doc_path in enumerate(documents, 1):
                print(f"  {i}. {doc_path}")  # noqa: T201
        else:
            print("  (Unable to list document sources)")  # noqa: T201

    except (OSError, ValueError, RuntimeError) as e:
        print(f"✗ Error listing documents: {e}")  # noqa: T201
        sys.exit(1)


def clear_knowledge_base() -> None:
    """Clear all documents from the knowledge base."""
    try:
        kb = KnowledgeBase()
        info = kb.get_info()

        if not info["exists"] or info["count"] == 0:
            print("Knowledge base is already empty.")  # noqa: T201
            return

        print(f"Knowledge base contains {info['count']} chunks.")  # noqa: T201
        response = input("Are you sure you want to clear the knowledge base? (yes/no): ").strip().lower()

        if response == "yes":
            kb.clear()
            print("✓ Knowledge base cleared successfully.")  # noqa: T201
        else:
            print("Operation cancelled.")  # noqa: T201

    except (OSError, ValueError, RuntimeError) as e:
        print(f"✗ Error clearing knowledge base: {e}")  # noqa: T201
        sys.exit(1)
