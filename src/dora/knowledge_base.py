"""Knowledge base management module."""

from pathlib import Path
from typing import Any

from dora.document import DocumentProcessor
from dora.vectorstore import VectorStore


class KnowledgeBase:
    """Manage the knowledge base: add documents, query, and manage the vector store."""

    def __init__(
        self,
        kb_directory: str | Path = ".dora/kb",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        embedding_model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
    ) -> None:
        """Initialize the knowledge base.

        Parameters
        ----------
        kb_directory : str | Path, optional
            Directory to store the knowledge base, by default ".dora/kb"
        chunk_size : int, optional
            Size of each text chunk in characters, by default 1000
        chunk_overlap : int, optional
            Number of characters to overlap between chunks, by default 200
        embedding_model_name : str, optional
            Name of the HuggingFace embedding model, by default
            "paraphrase-multilingual-MiniLM-L12-v2"
        """
        self.kb_directory = Path(kb_directory)
        self.document_processor = DocumentProcessor(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self.vectorstore = VectorStore(
            persist_directory=self.kb_directory,
            embedding_model_name=embedding_model_name,
        )

    def add_document(self, file_path: str | Path) -> int:
        """Add a document to the knowledge base.

        Parameters
        ----------
        file_path : str | Path
            Path to the PDF file to add

        Returns
        -------
        int
            Number of chunks added to the knowledge base

        Raises
        ------
        RuntimeError
            If document processing fails or no content is extracted
        """
        file_path = Path(file_path)

        # Load and split the document
        chunks = self.document_processor.load_pdf(file_path)

        if not chunks:
            msg = f"No content extracted from {file_path}"
            raise RuntimeError(msg)

        # Add to vector store
        self.vectorstore.add_documents(chunks)

        return len(chunks)

    def search(self, query: str, k: int = 4) -> list[Any]:
        """Search the knowledge base for relevant documents.

        Parameters
        ----------
        query : str
            Search query
        k : int, optional
            Number of documents to retrieve, by default 4

        Returns
        -------
        list[Any]
            List of relevant document chunks
        """
        return self.vectorstore.similarity_search(query, k=k)

    def get_retriever(self, **kwargs: Any) -> Any:
        """Get a retriever for the knowledge base.

        Parameters
        ----------
        **kwargs : Any
            Additional arguments for the retriever

        Returns
        -------
        Any
            LangChain retriever object
        """
        return self.vectorstore.as_retriever(**kwargs)

    def clear(self) -> None:
        """Clear all documents from the knowledge base."""
        self.vectorstore.delete_collection()
        self.vectorstore = VectorStore(
            persist_directory=self.kb_directory,
            embedding_model_name=self.vectorstore.embeddings.model_name,
        )

    def get_info(self) -> dict[str, Any]:
        """Get information about the knowledge base.

        Returns
        -------
        dict[str, Any]
            Information about the knowledge base including document count
        """
        return self.vectorstore.get_collection_info()

    def list_documents(self) -> list[str]:
        """List unique document sources in the knowledge base.

        Returns
        -------
        list[str]
            List of unique document file paths
        """
        info = self.get_info()
        if not info["exists"] or info["count"] == 0:
            return []

        # Get all documents to extract unique sources
        # Note: This is a simple implementation. For large knowledge bases,
        # you might want to store metadata separately
        try:
            # Search with a generic query to get some documents
            docs = self.vectorstore.similarity_search("", k=min(100, info["count"]))
            sources = set()
            for doc in docs:
                if hasattr(doc, "metadata") and "source" in doc.metadata:
                    sources.add(doc.metadata["source"])
            return sorted(sources)
        except (AttributeError, KeyError, TypeError):
            return []
