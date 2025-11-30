"""Test package initialization."""

from dora import LocalLLM, __all__, __version__


def test_version() -> None:
    """Test that __version__ is defined."""
    assert __version__ == "0.0.1"


def test_all() -> None:
    """Test that __all__ is defined correctly."""
    assert __all__ == ["LocalLLM"]
    assert "LocalLLM" in __all__


def test_import_local_llm() -> None:
    """Test that LocalLLM can be imported from dora."""
    assert LocalLLM is not None
    assert hasattr(LocalLLM, "__init__")
    assert hasattr(LocalLLM, "invoke")
