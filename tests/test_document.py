"""Tests for document processing module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from dora.document import DocumentProcessor


def test_document_processor_initialization() -> None:
    """Test that DocumentProcessor can be initialized."""
    processor = DocumentProcessor(chunk_size=500, chunk_overlap=100)
    assert processor.chunk_size == 500
    assert processor.chunk_overlap == 100
    assert processor.text_splitter is not None


def test_document_processor_default_initialization() -> None:
    """Test that DocumentProcessor uses default values."""
    processor = DocumentProcessor()
    assert processor.chunk_size == 1000
    assert processor.chunk_overlap == 200


@patch("dora.document.PyPDFLoader")
def test_load_pdf_success(mock_loader_class: MagicMock) -> None:
    """Test successful PDF loading."""
    # Mock document
    mock_doc = MagicMock()
    mock_doc.page_content = "This is a test document. " * 50  # Long enough to split
    mock_doc.metadata = {}

    # Mock loader
    mock_loader = MagicMock()
    mock_loader.load.return_value = [mock_doc]
    mock_loader_class.return_value = mock_loader

    processor = DocumentProcessor(chunk_size=100, chunk_overlap=20)

    # Create a temporary file path (doesn't need to exist for this test)
    file_path = Path("test.pdf")

    with patch.object(Path, "exists", return_value=True), patch.object(Path, "suffix", new_callable=lambda: ".pdf"):
        chunks = processor.load_pdf(file_path)

        assert len(chunks) > 0
        assert all(hasattr(chunk, "page_content") for chunk in chunks)
        assert all(hasattr(chunk, "metadata") for chunk in chunks)


def test_load_pdf_file_not_found() -> None:
    """Test that FileNotFoundError is raised for non-existent files."""
    processor = DocumentProcessor()
    file_path = Path("nonexistent.pdf")

    with pytest.raises(FileNotFoundError):
        processor.load_pdf(file_path)


def test_load_pdf_invalid_format() -> None:
    """Test that ValueError is raised for non-PDF files."""
    processor = DocumentProcessor()
    file_path = Path("test.txt")

    with patch.object(Path, "exists", return_value=True), pytest.raises(ValueError, match="must be a PDF"):
        processor.load_pdf(file_path)


@patch("dora.document.PyPDFLoader")
def test_load_pdf_runtime_error(mock_loader_class: MagicMock) -> None:
    """Test that RuntimeError is raised when PDF loading fails."""
    mock_loader = MagicMock()
    mock_loader.load.side_effect = Exception("PDF parsing failed")
    mock_loader_class.return_value = mock_loader

    processor = DocumentProcessor()
    file_path = Path("test.pdf")

    with (
        patch.object(Path, "exists", return_value=True),
        pytest.raises(RuntimeError, match="Failed to load PDF"),
    ):
        processor.load_pdf(file_path)


@patch("dora.document.PyPDFLoader")
def test_load_pdf_adds_source_metadata(mock_loader_class: MagicMock) -> None:
    """Test that source metadata is added when not present."""
    # Mock document without source metadata
    mock_doc = MagicMock()
    mock_doc.page_content = "Short test content"
    mock_doc.metadata = {}  # No source metadata

    mock_loader = MagicMock()
    mock_loader.load.return_value = [mock_doc]
    mock_loader_class.return_value = mock_loader

    processor = DocumentProcessor(chunk_size=1000, chunk_overlap=0)
    file_path = Path("test.pdf")

    with patch.object(Path, "exists", return_value=True):
        chunks = processor.load_pdf(file_path)

        assert len(chunks) > 0
        # Check that source and file_name metadata were added
        for chunk in chunks:
            assert "source" in chunk.metadata
            assert "file_name" in chunk.metadata
            assert chunk.metadata["file_name"] == "test.pdf"


@patch("dora.document.PyPDFLoader")
def test_load_pdf_preserves_existing_source_metadata(mock_loader_class: MagicMock) -> None:
    """Test that existing source metadata is preserved."""
    # Mock document with existing source metadata
    mock_doc = MagicMock()
    mock_doc.page_content = "Short test content"
    mock_doc.metadata = {"source": "/original/path/doc.pdf"}  # Already has source

    mock_loader = MagicMock()
    mock_loader.load.return_value = [mock_doc]
    mock_loader_class.return_value = mock_loader

    processor = DocumentProcessor(chunk_size=1000, chunk_overlap=0)
    file_path = Path("test.pdf")

    with patch.object(Path, "exists", return_value=True):
        chunks = processor.load_pdf(file_path)

        assert len(chunks) > 0
        # Check that original source is preserved
        for chunk in chunks:
            assert chunk.metadata["source"] == "/original/path/doc.pdf"
            assert chunk.metadata["file_name"] == "test.pdf"
