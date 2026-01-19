"""Local LLM implementation using Ollama and LangChain."""

from typing import Any

from langchain_ollama import OllamaLLM

from dora.knowledge_base import KnowledgeBase
from dora.rag import RAGChain


class LocalLLM:
    """Wrapper class for local LLM using Ollama with optional RAG support."""

    def __init__(
        self,
        model_name: str = "llama3.2",
        use_rag: bool = False,
        knowledge_base: KnowledgeBase | None = None,
        rag_k: int = 4,
    ) -> None:
        """Initialize the local LLM.

        Parameters
        ----------
        model_name : str, optional
            Name of the Ollama model to use, by default "llama3.2"
        use_rag : bool, optional
            Whether to use RAG for answering queries, by default False
        knowledge_base : KnowledgeBase | None, optional
            Knowledge base instance for RAG, by default None
        rag_k : int, optional
            Number of documents to retrieve for RAG, by default 4

        Raises
        ------
        ConnectionError
            If Ollama is not running or the model is not available
        ValueError
            If use_rag is True but knowledge_base is None
        """
        self.model_name = model_name
        self.use_rag = use_rag
        self.knowledge_base = knowledge_base
        self.rag_k = rag_k

        if use_rag and knowledge_base is None:
            msg = "knowledge_base must be provided when use_rag=True"
            raise ValueError(msg)

        try:
            self.llm = OllamaLLM(model=model_name)
        except Exception as e:
            error_msg = (
                f"Failed to connect to Ollama or model '{model_name}' is not available. "
                f"Make sure Ollama is running and the model is pulled: ollama pull {model_name}"
            )
            raise ConnectionError(error_msg) from e

        # Initialize RAG chain if RAG is enabled
        self.rag_chain: RAGChain | None = None
        if use_rag and knowledge_base is not None:
            self.rag_chain = RAGChain(
                knowledge_base=knowledge_base,
                llm=self.llm,
                k=rag_k,
            )

    def invoke(self, prompt: str) -> str:
        """Generate a response from the LLM.

        If RAG is enabled, uses the knowledge base to augment the response.

        Parameters
        ----------
        prompt : str
            Input prompt for the LLM

        Returns
        -------
        str
            Generated response from the LLM

        Raises
        ------
        RuntimeError
            If the LLM fails to generate a response
        """
        try:
            if self.use_rag and self.rag_chain is not None:
                # Use RAG chain
                response = self.rag_chain.get_answer(prompt)
            else:
                # Direct LLM query
                response = self.llm.invoke(prompt)
        except Exception as e:
            error_msg = f"Failed to generate response: {e}"
            raise RuntimeError(error_msg) from e
        else:
            return response

    def invoke_with_sources(self, prompt: str) -> dict[str, Any]:
        """Generate a response with source documents (RAG only).

        Parameters
        ----------
        prompt : str
            Input prompt for the LLM

        Returns
        -------
        dict[str, Any]
            Dictionary containing:
            - "result": The generated answer
            - "source_documents": List of source documents used

        Raises
        ------
        RuntimeError
            If the LLM fails to generate a response
        ValueError
            If RAG is not enabled
        """
        if not self.use_rag or self.rag_chain is None:
            msg = "RAG must be enabled to use invoke_with_sources"
            raise ValueError(msg)

        try:
            return self.rag_chain.invoke(prompt)
        except Exception as e:
            error_msg = f"Failed to generate response: {e}"
            raise RuntimeError(error_msg) from e
