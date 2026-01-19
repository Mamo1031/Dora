"""Test CLI interface for Dora."""

from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from dora.cli import (
    _initialize_knowledge_base,  # noqa: PLC2701
    _initialize_llm,  # noqa: PLC2701
    add_document,
    clear_knowledge_base,
    interactive,
    list_documents,
    run_test_llm,
)


@patch("dora.cli.LocalLLM")
@patch("sys.stdout", new_callable=StringIO)
@patch("sys.exit")
def test_cli_success(
    mock_exit: MagicMock,
    mock_stdout: StringIO,
    mock_local_llm: MagicMock,
) -> None:
    """Test CLI success case."""
    mock_llm_instance = MagicMock()
    mock_llm_instance.invoke.return_value = "Test response"
    mock_local_llm.return_value = mock_llm_instance

    run_test_llm()

    output = mock_stdout.getvalue()
    assert "Initializing Local LLM" in output
    assert "LLM initialized successfully" in output
    assert "Test prompt" in output
    assert "Response:" in output
    assert "Test response" in output
    # sys.exit should not be called on success
    mock_exit.assert_not_called()


@patch("dora.cli.LocalLLM")
@patch("sys.stdout", new_callable=StringIO)
@patch("sys.exit")
def test_cli_connection_error(
    mock_exit: MagicMock,
    mock_stdout: StringIO,
    mock_local_llm: MagicMock,
) -> None:
    """Test CLI when ConnectionError occurs."""
    mock_local_llm.side_effect = ConnectionError("Connection failed")
    # Make sys.exit raise SystemExit to stop execution
    mock_exit.side_effect = SystemExit(1)

    with pytest.raises(SystemExit):
        run_test_llm()

    output = mock_stdout.getvalue()
    assert "Error:" in output
    assert "Connection failed" in output
    assert "Please ensure:" in output
    assert "Ollama is installed and running" in output
    mock_exit.assert_called_once_with(1)


@patch("dora.cli.LocalLLM")
@patch("sys.stdout", new_callable=StringIO)
@patch("sys.exit")
def test_cli_runtime_error(
    mock_exit: MagicMock,
    mock_stdout: StringIO,
    mock_local_llm: MagicMock,
) -> None:
    """Test CLI when RuntimeError occurs during invoke."""
    mock_llm_instance = MagicMock()
    mock_llm_instance.invoke.side_effect = RuntimeError("Invoke failed")
    mock_local_llm.return_value = mock_llm_instance
    # Make sys.exit raise SystemExit to stop execution
    mock_exit.side_effect = SystemExit(1)

    with pytest.raises(SystemExit):
        run_test_llm()

    output = mock_stdout.getvalue()
    assert "LLM initialized successfully" in output
    assert "Error generating response" in output
    assert "Invoke failed" in output
    mock_exit.assert_called_once_with(1)


@patch("dora.cli.LocalLLM")
@patch("sys.stdout", new_callable=StringIO)
@patch("builtins.input")
@patch("sys.exit")
def test_interactive_success(
    mock_exit: MagicMock,
    mock_input: MagicMock,
    mock_stdout: StringIO,
    mock_local_llm: MagicMock,
) -> None:
    """Test interactive mode success case."""
    mock_llm_instance = MagicMock()
    mock_llm_instance.invoke.return_value = "Test response"
    mock_local_llm.return_value = mock_llm_instance
    mock_input.side_effect = ["Hello", "exit"]

    interactive()

    output = mock_stdout.getvalue()
    assert "Initializing Local LLM" in output
    assert "LLM initialized successfully" in output
    assert "Interactive mode started" in output
    assert "Generating response..." in output
    assert "Assistant: Test response" in output
    assert "Goodbye!" in output
    mock_exit.assert_not_called()
    # Verify input was called (at least "Hello" and "exit")
    assert mock_input.call_count >= 2


@patch("dora.cli.LocalLLM")
@patch("sys.stdout", new_callable=StringIO)
@patch("sys.exit")
def test_interactive_connection_error(
    mock_exit: MagicMock,
    mock_stdout: StringIO,
    mock_local_llm: MagicMock,
) -> None:
    """Test interactive mode when ConnectionError occurs."""
    mock_local_llm.side_effect = ConnectionError("Connection failed")
    mock_exit.side_effect = SystemExit(1)

    with pytest.raises(SystemExit):
        interactive()

    output = mock_stdout.getvalue()
    assert "Error:" in output
    assert "Connection failed" in output
    assert "Please ensure:" in output
    mock_exit.assert_called_once_with(1)


@patch("dora.cli.LocalLLM")
@patch("sys.stdout", new_callable=StringIO)
@patch("builtins.input")
def test_interactive_empty_input(
    mock_input: MagicMock,
    mock_stdout: StringIO,
    mock_local_llm: MagicMock,
) -> None:
    """Test interactive mode with empty input."""
    mock_llm_instance = MagicMock()
    mock_llm_instance.invoke.return_value = "Response"
    mock_local_llm.return_value = mock_llm_instance
    mock_input.side_effect = ["", "  ", "Hello", "exit"]

    interactive()

    output = mock_stdout.getvalue()
    assert "Interactive mode started" in output
    assert "Assistant: Response" in output
    assert "Goodbye!" in output


@patch("dora.cli.LocalLLM")
@patch("sys.stdout", new_callable=StringIO)
@patch("builtins.input")
def test_interactive_quit_command(
    mock_input: MagicMock,
    mock_stdout: StringIO,
    mock_local_llm: MagicMock,
) -> None:
    """Test interactive mode with quit command."""
    mock_llm_instance = MagicMock()
    mock_local_llm.return_value = mock_llm_instance
    mock_input.side_effect = ["quit"]

    interactive()

    output = mock_stdout.getvalue()
    assert "Interactive mode started" in output
    assert "Goodbye!" in output
    mock_llm_instance.invoke.assert_not_called()


@patch("dora.cli.LocalLLM")
@patch("sys.stdout", new_callable=StringIO)
@patch("builtins.input")
def test_interactive_runtime_error(
    mock_input: MagicMock,
    mock_stdout: StringIO,
    mock_local_llm: MagicMock,
) -> None:
    """Test interactive mode when RuntimeError occurs during invoke."""
    mock_llm_instance = MagicMock()
    mock_llm_instance.invoke.side_effect = RuntimeError("Invoke failed")
    mock_local_llm.return_value = mock_llm_instance
    mock_input.side_effect = ["Hello", "exit"]

    interactive()

    output = mock_stdout.getvalue()
    assert "Interactive mode started" in output
    assert "Error generating response" in output
    assert "Invoke failed" in output
    assert "Goodbye!" in output


@patch("dora.cli.LocalLLM")
@patch("sys.stdout", new_callable=StringIO)
@patch("builtins.input")
def test_interactive_keyboard_interrupt(
    mock_input: MagicMock,
    mock_stdout: StringIO,
    mock_local_llm: MagicMock,
) -> None:
    """Test interactive mode with KeyboardInterrupt."""
    mock_llm_instance = MagicMock()
    mock_local_llm.return_value = mock_llm_instance
    mock_input.side_effect = KeyboardInterrupt()

    interactive()

    output = mock_stdout.getvalue()
    assert "Interactive mode started" in output
    assert "Goodbye!" in output


@patch("dora.cli.LocalLLM")
@patch("sys.stdout", new_callable=StringIO)
@patch("builtins.input")
def test_interactive_eof_error(
    mock_input: MagicMock,
    mock_stdout: StringIO,
    mock_local_llm: MagicMock,
) -> None:
    """Test interactive mode with EOFError."""
    mock_llm_instance = MagicMock()
    mock_local_llm.return_value = mock_llm_instance
    mock_input.side_effect = EOFError()

    interactive()

    output = mock_stdout.getvalue()
    assert "Interactive mode started" in output
    assert "Goodbye!" in output


# ============================================================================
# add_document tests
# ============================================================================


@patch("dora.cli.KnowledgeBase")
@patch("sys.stdout", new_callable=StringIO)
@patch("sys.argv", ["dora", "add-doc", "/path/to/document.pdf"])
@patch("sys.exit")
def test_add_document_success(
    mock_exit: MagicMock,
    mock_stdout: StringIO,
    mock_knowledge_base: MagicMock,
) -> None:
    """Test add_document success case."""
    mock_kb_instance = MagicMock()
    mock_kb_instance.add_document.return_value = 10
    mock_knowledge_base.return_value = mock_kb_instance

    with patch.object(Path, "exists", return_value=True):
        add_document()

    output = mock_stdout.getvalue()
    assert "Adding document:" in output
    assert "Processing document..." in output
    assert "Document added successfully (10 chunks)" in output
    mock_exit.assert_not_called()


@patch("sys.stdout", new_callable=StringIO)
@patch("sys.argv", ["dora", "add-doc"])
@patch("sys.exit")
def test_add_document_missing_args(
    mock_exit: MagicMock,
    mock_stdout: StringIO,
) -> None:
    """Test add_document when file path is missing."""
    mock_exit.side_effect = SystemExit(1)

    with pytest.raises(SystemExit):
        add_document()

    output = mock_stdout.getvalue()
    assert "Usage: dora add-doc <file_path>" in output
    mock_exit.assert_called_once_with(1)


@patch("sys.stdout", new_callable=StringIO)
@patch("sys.argv", ["dora", "add-doc", "/path/to/nonexistent.pdf"])
@patch("sys.exit")
def test_add_document_file_not_found(
    mock_exit: MagicMock,
    mock_stdout: StringIO,
) -> None:
    """Test add_document when file does not exist."""
    mock_exit.side_effect = SystemExit(1)

    with pytest.raises(SystemExit):
        add_document()

    output = mock_stdout.getvalue()
    assert "Error: File not found:" in output
    mock_exit.assert_called_once_with(1)


@patch("sys.stdout", new_callable=StringIO)
@patch("sys.argv", ["dora", "add-doc", "/path/to/document.txt"])
@patch("sys.exit")
def test_add_document_not_pdf(
    mock_exit: MagicMock,
    mock_stdout: StringIO,
) -> None:
    """Test add_document when file is not a PDF."""
    mock_exit.side_effect = SystemExit(1)

    with patch.object(Path, "exists", return_value=True), pytest.raises(SystemExit):
        add_document()

    output = mock_stdout.getvalue()
    assert "Error: Only PDF files are supported:" in output
    mock_exit.assert_called_once_with(1)


@patch("dora.cli.KnowledgeBase")
@patch("sys.stdout", new_callable=StringIO)
@patch("sys.argv", ["dora", "add-doc", "/path/to/document.pdf"])
@patch("sys.exit")
def test_add_document_error(
    mock_exit: MagicMock,
    mock_stdout: StringIO,
    mock_knowledge_base: MagicMock,
) -> None:
    """Test add_document when KnowledgeBase raises an error."""
    mock_kb_instance = MagicMock()
    mock_kb_instance.add_document.side_effect = RuntimeError("Processing failed")
    mock_knowledge_base.return_value = mock_kb_instance
    mock_exit.side_effect = SystemExit(1)

    with patch.object(Path, "exists", return_value=True), pytest.raises(SystemExit):
        add_document()

    output = mock_stdout.getvalue()
    assert "Error adding document:" in output
    assert "Processing failed" in output
    mock_exit.assert_called_once_with(1)


# ============================================================================
# list_documents tests
# ============================================================================


@patch("dora.cli.KnowledgeBase")
@patch("sys.stdout", new_callable=StringIO)
def test_list_documents_with_documents(
    mock_stdout: StringIO,
    mock_knowledge_base: MagicMock,
) -> None:
    """Test list_documents when documents exist."""
    mock_kb_instance = MagicMock()
    mock_kb_instance.get_info.return_value = {"exists": True, "count": 10}
    mock_kb_instance.list_documents.return_value = [
        "/path/to/doc1.pdf",
        "/path/to/doc2.pdf",
    ]
    mock_knowledge_base.return_value = mock_kb_instance

    list_documents()

    output = mock_stdout.getvalue()
    assert "Knowledge base contains 10 chunks." in output
    assert "Documents:" in output
    assert "1. /path/to/doc1.pdf" in output
    assert "2. /path/to/doc2.pdf" in output


@patch("dora.cli.KnowledgeBase")
@patch("sys.stdout", new_callable=StringIO)
def test_list_documents_empty(
    mock_stdout: StringIO,
    mock_knowledge_base: MagicMock,
) -> None:
    """Test list_documents when no documents exist."""
    mock_kb_instance = MagicMock()
    mock_kb_instance.get_info.return_value = {"exists": True, "count": 0}
    mock_knowledge_base.return_value = mock_kb_instance

    list_documents()

    output = mock_stdout.getvalue()
    assert "No documents in knowledge base." in output
    assert "Use 'dora add-doc <file>' to add documents." in output


@patch("dora.cli.KnowledgeBase")
@patch("sys.stdout", new_callable=StringIO)
def test_list_documents_not_exists(
    mock_stdout: StringIO,
    mock_knowledge_base: MagicMock,
) -> None:
    """Test list_documents when knowledge base doesn't exist."""
    mock_kb_instance = MagicMock()
    mock_kb_instance.get_info.return_value = {"exists": False, "count": 0}
    mock_knowledge_base.return_value = mock_kb_instance

    list_documents()

    output = mock_stdout.getvalue()
    assert "No documents in knowledge base." in output


@patch("dora.cli.KnowledgeBase")
@patch("sys.stdout", new_callable=StringIO)
def test_list_documents_empty_document_list(
    mock_stdout: StringIO,
    mock_knowledge_base: MagicMock,
) -> None:
    """Test list_documents when document list is empty but count > 0."""
    mock_kb_instance = MagicMock()
    mock_kb_instance.get_info.return_value = {"exists": True, "count": 10}
    mock_kb_instance.list_documents.return_value = []
    mock_knowledge_base.return_value = mock_kb_instance

    list_documents()

    output = mock_stdout.getvalue()
    assert "Knowledge base contains 10 chunks." in output
    assert "(Unable to list document sources)" in output


@patch("dora.cli.KnowledgeBase")
@patch("sys.stdout", new_callable=StringIO)
@patch("sys.exit")
def test_list_documents_error(
    mock_exit: MagicMock,
    mock_stdout: StringIO,
    mock_knowledge_base: MagicMock,
) -> None:
    """Test list_documents when an error occurs."""
    mock_knowledge_base.side_effect = RuntimeError("Database error")
    mock_exit.side_effect = SystemExit(1)

    with pytest.raises(SystemExit):
        list_documents()

    output = mock_stdout.getvalue()
    assert "Error listing documents:" in output
    assert "Database error" in output
    mock_exit.assert_called_once_with(1)


# ============================================================================
# clear_knowledge_base tests
# ============================================================================


@patch("dora.cli.KnowledgeBase")
@patch("sys.stdout", new_callable=StringIO)
@patch("builtins.input")
def test_clear_knowledge_base_confirmed(
    mock_input: MagicMock,
    mock_stdout: StringIO,
    mock_knowledge_base: MagicMock,
) -> None:
    """Test clear_knowledge_base when user confirms."""
    mock_kb_instance = MagicMock()
    mock_kb_instance.get_info.return_value = {"exists": True, "count": 10}
    mock_knowledge_base.return_value = mock_kb_instance
    mock_input.return_value = "yes"

    clear_knowledge_base()

    output = mock_stdout.getvalue()
    assert "Knowledge base contains 10 chunks." in output
    assert "Knowledge base cleared successfully." in output
    mock_kb_instance.clear.assert_called_once()


@patch("dora.cli.KnowledgeBase")
@patch("sys.stdout", new_callable=StringIO)
@patch("builtins.input")
def test_clear_knowledge_base_cancelled(
    mock_input: MagicMock,
    mock_stdout: StringIO,
    mock_knowledge_base: MagicMock,
) -> None:
    """Test clear_knowledge_base when user cancels."""
    mock_kb_instance = MagicMock()
    mock_kb_instance.get_info.return_value = {"exists": True, "count": 10}
    mock_knowledge_base.return_value = mock_kb_instance
    mock_input.return_value = "no"

    clear_knowledge_base()

    output = mock_stdout.getvalue()
    assert "Knowledge base contains 10 chunks." in output
    assert "Operation cancelled." in output
    mock_kb_instance.clear.assert_not_called()


@patch("dora.cli.KnowledgeBase")
@patch("sys.stdout", new_callable=StringIO)
def test_clear_knowledge_base_already_empty(
    mock_stdout: StringIO,
    mock_knowledge_base: MagicMock,
) -> None:
    """Test clear_knowledge_base when already empty."""
    mock_kb_instance = MagicMock()
    mock_kb_instance.get_info.return_value = {"exists": True, "count": 0}
    mock_knowledge_base.return_value = mock_kb_instance

    clear_knowledge_base()

    output = mock_stdout.getvalue()
    assert "Knowledge base is already empty." in output
    mock_kb_instance.clear.assert_not_called()


@patch("dora.cli.KnowledgeBase")
@patch("sys.stdout", new_callable=StringIO)
def test_clear_knowledge_base_not_exists(
    mock_stdout: StringIO,
    mock_knowledge_base: MagicMock,
) -> None:
    """Test clear_knowledge_base when knowledge base doesn't exist."""
    mock_kb_instance = MagicMock()
    mock_kb_instance.get_info.return_value = {"exists": False, "count": 0}
    mock_knowledge_base.return_value = mock_kb_instance

    clear_knowledge_base()

    output = mock_stdout.getvalue()
    assert "Knowledge base is already empty." in output
    mock_kb_instance.clear.assert_not_called()


@patch("dora.cli.KnowledgeBase")
@patch("sys.stdout", new_callable=StringIO)
@patch("sys.exit")
def test_clear_knowledge_base_error(
    mock_exit: MagicMock,
    mock_stdout: StringIO,
    mock_knowledge_base: MagicMock,
) -> None:
    """Test clear_knowledge_base when an error occurs."""
    mock_knowledge_base.side_effect = RuntimeError("Database error")
    mock_exit.side_effect = SystemExit(1)

    with pytest.raises(SystemExit):
        clear_knowledge_base()

    output = mock_stdout.getvalue()
    assert "Error clearing knowledge base:" in output
    assert "Database error" in output
    mock_exit.assert_called_once_with(1)


# ============================================================================
# _initialize_knowledge_base tests
# ============================================================================


@patch("dora.cli.KnowledgeBase")
@patch("sys.stdout", new_callable=StringIO)
def test_initialize_knowledge_base_with_documents(
    mock_stdout: StringIO,
    mock_knowledge_base: MagicMock,
) -> None:
    """Test _initialize_knowledge_base when documents exist."""
    mock_kb_instance = MagicMock()
    mock_kb_instance.get_info.return_value = {"exists": True, "count": 10}
    mock_knowledge_base.return_value = mock_kb_instance

    kb, use_rag = _initialize_knowledge_base()

    assert kb is mock_kb_instance
    assert use_rag is True
    output = mock_stdout.getvalue()
    assert "Knowledge base found (10 chunks)" in output
    assert "RAG mode enabled" in output


@patch("dora.cli.KnowledgeBase")
@patch("sys.stdout", new_callable=StringIO)
def test_initialize_knowledge_base_empty(
    mock_stdout: StringIO,
    mock_knowledge_base: MagicMock,
) -> None:
    """Test _initialize_knowledge_base when no documents exist."""
    mock_kb_instance = MagicMock()
    mock_kb_instance.get_info.return_value = {"exists": True, "count": 0}
    mock_knowledge_base.return_value = mock_kb_instance

    kb, use_rag = _initialize_knowledge_base()

    assert kb is mock_kb_instance
    assert use_rag is False
    output = mock_stdout.getvalue()
    assert "No documents in knowledge base" in output
    assert "RAG disabled" in output


@patch("dora.cli.KnowledgeBase")
@patch("sys.stdout", new_callable=StringIO)
def test_initialize_knowledge_base_error(
    mock_stdout: StringIO,
    mock_knowledge_base: MagicMock,
) -> None:
    """Test _initialize_knowledge_base when initialization fails."""
    mock_knowledge_base.side_effect = RuntimeError("Init failed")

    kb, use_rag = _initialize_knowledge_base()

    assert kb is None
    assert use_rag is False
    output = mock_stdout.getvalue()
    assert "Could not initialize knowledge base" in output
    assert "RAG disabled" in output


# ============================================================================
# _initialize_llm tests
# ============================================================================


@patch("dora.cli.LocalLLM")
def test_initialize_llm_success(
    mock_local_llm: MagicMock,
) -> None:
    """Test _initialize_llm success case."""
    mock_llm_instance = MagicMock()
    mock_local_llm.return_value = mock_llm_instance
    mock_kb = MagicMock()

    result = _initialize_llm(use_rag=True, kb=mock_kb)

    assert result is mock_llm_instance
    mock_local_llm.assert_called_once_with(
        model_name="llama3.2",
        use_rag=True,
        knowledge_base=mock_kb,
    )


@patch("dora.cli.LocalLLM")
@patch("sys.stdout", new_callable=StringIO)
@patch("sys.exit")
def test_initialize_llm_value_error(
    mock_exit: MagicMock,
    mock_stdout: StringIO,
    mock_local_llm: MagicMock,
) -> None:
    """Test _initialize_llm when ValueError occurs."""
    mock_local_llm.side_effect = ValueError("Invalid configuration")
    mock_exit.side_effect = SystemExit(1)

    with pytest.raises(SystemExit):
        _initialize_llm(use_rag=False, kb=None)

    output = mock_stdout.getvalue()
    assert "Error:" in output
    assert "Invalid configuration" in output
    mock_exit.assert_called_once_with(1)
