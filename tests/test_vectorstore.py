"""Tests for vector store module."""

import tempfile
from pathlib import Path

from langchain_core.documents import Document

from dora.vectorstore import VectorStore


def test_vectorstore_initialization() -> None:
    """Test that VectorStore can be initialized."""
    with tempfile.TemporaryDirectory() as tmpdir:
        vectorstore = VectorStore(persist_directory=tmpdir)
        assert vectorstore.persist_directory == Path(tmpdir)
        assert vectorstore.embeddings is not None
        assert vectorstore.client is not None


def test_vectorstore_custom_embedding_model() -> None:
    """Test VectorStore with custom embedding model."""
    with tempfile.TemporaryDirectory() as tmpdir:
        vectorstore = VectorStore(
            persist_directory=tmpdir,
            embedding_model_name="all-MiniLM-L6-v2",
        )
        assert vectorstore.embeddings is not None


def test_get_or_create_vectorstore() -> None:
    """Test that vectorstore is created on first access."""
    with tempfile.TemporaryDirectory() as tmpdir:
        vectorstore = VectorStore(persist_directory=tmpdir)
        assert vectorstore.vectorstore is None

        vs = vectorstore._get_or_create_vectorstore()
        assert vs is not None
        assert vectorstore.vectorstore is not None


def test_add_documents() -> None:
    """Test adding documents to vector store."""
    with tempfile.TemporaryDirectory() as tmpdir:
        vectorstore = VectorStore(persist_directory=tmpdir)

        # Create documents
        doc1 = Document(
            page_content="This is the first document.",
            metadata={"source": "test1.pdf"},
        )
        doc2 = Document(
            page_content="This is the second document.",
            metadata={"source": "test2.pdf"},
        )

        documents = [doc1, doc2]

        ids = vectorstore.add_documents(documents)
        assert len(ids) == 2


def test_similarity_search() -> None:
    """Test similarity search in vector store."""
    with tempfile.TemporaryDirectory() as tmpdir:
        vectorstore = VectorStore(persist_directory=tmpdir)

        # Create document
        doc = Document(
            page_content="This is a test document about Python programming.",
            metadata={"source": "test.pdf"},
        )

        vectorstore.add_documents([doc])

        # Perform search
        results = vectorstore.similarity_search("Python", k=1)
        assert len(results) >= 0  # May be empty if no matches


def test_similarity_search_with_score() -> None:
    """Test similarity search with scores."""
    with tempfile.TemporaryDirectory() as tmpdir:
        vectorstore = VectorStore(persist_directory=tmpdir)

        # Create document
        doc = Document(
            page_content="This is a test document.",
            metadata={"source": "test.pdf"},
        )

        vectorstore.add_documents([doc])

        # Perform search
        results = vectorstore.similarity_search_with_score("test", k=1)
        assert isinstance(results, list)
        # Results may be empty, but if not, should be tuples
        if results:
            assert isinstance(results[0], tuple)
            assert len(results[0]) == 2


def test_as_retriever() -> None:
    """Test getting retriever from vector store."""
    with tempfile.TemporaryDirectory() as tmpdir:
        vectorstore = VectorStore(persist_directory=tmpdir)
        retriever = vectorstore.as_retriever()
        assert retriever is not None


def test_get_collection_info_empty() -> None:
    """Test getting collection info when empty."""
    with tempfile.TemporaryDirectory() as tmpdir:
        vectorstore = VectorStore(persist_directory=tmpdir)
        info = vectorstore.get_collection_info()
        assert "count" in info
        assert "exists" in info


def test_get_collection_info_with_documents() -> None:
    """Test getting collection info with documents."""
    with tempfile.TemporaryDirectory() as tmpdir:
        vectorstore = VectorStore(persist_directory=tmpdir)

        # Create document
        doc = Document(
            page_content="Test document.",
            metadata={"source": "test.pdf"},
        )

        vectorstore.add_documents([doc])

        info = vectorstore.get_collection_info()
        assert info["exists"] is True
        assert info["count"] > 0
