"""Vector store module for managing embeddings and ChromaDB."""

from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


class VectorStore:
    """Manage vector store using ChromaDB and sentence-transformers embeddings."""

    def __init__(
        self,
        persist_directory: str | Path,
        embedding_model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
    ) -> None:
        """Initialize the vector store.

        Parameters
        ----------
        persist_directory : str | Path
            Directory to persist the ChromaDB database
        embedding_model_name : str, optional
            Name of the HuggingFace embedding model, by default
            "paraphrase-multilingual-MiniLM-L12-v2"
        """
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(anonymized_telemetry=False),
        )

        # Initialize vector store (will be created on first use)
        self.vectorstore: Chroma | None = None

    def _get_or_create_vectorstore(self) -> Chroma:
        """Get or create the ChromaDB vector store.

        Returns
        -------
        Chroma
            The ChromaDB vector store instance
        """
        if self.vectorstore is None:
            self.vectorstore = Chroma(
                client=self.client,
                embedding_function=self.embeddings,
                persist_directory=str(self.persist_directory),
                collection_name="dora_kb",
            )
        return self.vectorstore

    def add_documents(self, documents: list[Any]) -> list[str]:
        """Add documents to the vector store.

        Parameters
        ----------
        documents : list[Any]
            List of LangChain Document objects

        Returns
        -------
        list[str]
            List of document IDs added to the vector store
        """
        vectorstore = self._get_or_create_vectorstore()
        # ChromaDB 0.4.x+ automatically persists, no need to call persist()
        return vectorstore.add_documents(documents)

    def similarity_search(
        self,
        query: str,
        k: int = 4,
    ) -> list[Any]:
        """Search for similar documents.

        Parameters
        ----------
        query : str
            Search query
        k : int, optional
            Number of documents to retrieve, by default 4

        Returns
        -------
        list[Any]
            List of similar documents
        """
        vectorstore = self._get_or_create_vectorstore()
        return vectorstore.similarity_search(query, k=k)

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
    ) -> list[tuple[Any, float]]:
        """Search for similar documents with similarity scores.

        Parameters
        ----------
        query : str
            Search query
        k : int, optional
            Number of documents to retrieve, by default 4

        Returns
        -------
        list[tuple[Any, float]]
            List of tuples containing documents and their similarity scores
        """
        vectorstore = self._get_or_create_vectorstore()
        return vectorstore.similarity_search_with_score(query, k=k)

    def as_retriever(self, **kwargs: Any) -> Any:
        """Get the vector store as a retriever.

        Parameters
        ----------
        **kwargs : Any
            Additional arguments for the retriever

        Returns
        -------
        Any
            LangChain retriever object
        """
        vectorstore = self._get_or_create_vectorstore()
        return vectorstore.as_retriever(**kwargs)

    def delete_collection(self) -> None:
        """Delete the entire collection."""
        if self.vectorstore is not None:
            self.client.delete_collection(name=self.vectorstore._collection.name)
            self.vectorstore = None

    def get_collection_info(self) -> dict[str, Any]:
        """Get information about the collection.

        Returns
        -------
        dict[str, Any]
            Collection information including count of documents
        """
        try:
            vectorstore = self._get_or_create_vectorstore()
            collection = vectorstore._collection
            count = collection.count()

            return {
                "count": count,
                "exists": True,
                "collection_name": collection.name,
            }
        except (AttributeError, RuntimeError, ValueError):
            # If collection doesn't exist or can't be accessed
            return {"count": 0, "exists": False}
