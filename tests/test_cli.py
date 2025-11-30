"""Test CLI interface for Dora."""

from io import StringIO
from unittest.mock import MagicMock, patch

import pytest

from dora.cli import interactive, run_test_llm


@patch("dora.cli.LocalLLM")
@patch("sys.stdout", new_callable=StringIO)
@patch("sys.exit")
def test_cli_success(
    mock_exit: MagicMock, mock_stdout: StringIO, mock_local_llm: MagicMock,
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
    mock_exit: MagicMock, mock_stdout: StringIO, mock_local_llm: MagicMock,
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
    mock_exit: MagicMock, mock_stdout: StringIO, mock_local_llm: MagicMock,
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
    assert mock_input.call_count >= 2  # noqa: PLR2004


@patch("dora.cli.LocalLLM")
@patch("sys.stdout", new_callable=StringIO)
@patch("sys.exit")
def test_interactive_connection_error(
    mock_exit: MagicMock, mock_stdout: StringIO, mock_local_llm: MagicMock,
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
    mock_input: MagicMock, mock_stdout: StringIO, mock_local_llm: MagicMock,
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
    mock_input: MagicMock, mock_stdout: StringIO, mock_local_llm: MagicMock,
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
    mock_input: MagicMock, mock_stdout: StringIO, mock_local_llm: MagicMock,
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
    mock_input: MagicMock, mock_stdout: StringIO, mock_local_llm: MagicMock,
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
    mock_input: MagicMock, mock_stdout: StringIO, mock_local_llm: MagicMock,
) -> None:
    """Test interactive mode with EOFError."""
    mock_llm_instance = MagicMock()
    mock_local_llm.return_value = mock_llm_instance
    mock_input.side_effect = EOFError()

    interactive()

    output = mock_stdout.getvalue()
    assert "Interactive mode started" in output
    assert "Goodbye!" in output
