"""RAG (Retrieval-Augmented Generation) chain implementation."""

import time
from typing import Any

from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

from dora.knowledge_base import KnowledgeBase


class RAGChain:
    """RAG chain that combines retrieval and generation."""

    def __init__(
        self,
        knowledge_base: KnowledgeBase,
        llm: OllamaLLM,
        k: int = 4,
    ) -> None:
        """Initialize the RAG chain.

        Parameters
        ----------
        knowledge_base : KnowledgeBase
            Knowledge base instance
        llm : OllamaLLM
            Language model instance
        k : int, optional
            Number of documents to retrieve, by default 4
        """
        self.knowledge_base = knowledge_base
        self.llm = llm
        self.k = k

        # Create prompt template for RAG
        prompt_template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context:
{context}

Question: {question}

Answer:"""

        self.prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"],
        )

        # Create retriever
        self.retriever = knowledge_base.get_retriever(search_kwargs={"k": k})

    def invoke(self, query: str) -> dict[str, Any]:
        """Invoke the RAG chain with a query.

        Parameters
        ----------
        query : str
            User query

        Returns
        -------
        dict[str, Any]
            Dictionary containing:
            - "result": The generated answer
            - "source_documents": List of source documents used
            - "performance": Dictionary with timing information:
                - "retrieval_time": Time spent on document retrieval (seconds)
                - "generation_time": Time spent on LLM generation (seconds)
                - "total_time": Total time for RAG chain (seconds)
        """
        # Measure retrieval time
        retrieval_start = time.time()
        source_documents = self.retriever.invoke(query)
        retrieval_time = time.time() - retrieval_start

        # Build context from documents
        context = "\n\n".join([doc.page_content for doc in source_documents])

        # Create prompt with context
        formatted_prompt = self.prompt.format(context=context, question=query)

        # Measure generation time
        generation_start = time.time()
        answer = self.llm.invoke(formatted_prompt)
        generation_time = time.time() - generation_start

        total_time = retrieval_time + generation_time

        return {
            "result": answer,
            "source_documents": source_documents,
            "performance": {
                "retrieval_time": retrieval_time,
                "generation_time": generation_time,
                "total_time": total_time,
            },
        }

    def get_answer(self, query: str) -> str:
        """Get only the answer text from the RAG chain.

        Parameters
        ----------
        query : str
            User query

        Returns
        -------
        str
            The generated answer
        """
        result = self.invoke(query)
        return result.get("result", "")

    def get_answer_with_performance(self, query: str) -> tuple[str, dict[str, float]]:
        """Get answer text and performance metrics from the RAG chain.

        Parameters
        ----------
        query : str
            User query

        Returns
        -------
        tuple[str, dict[str, float]]
            Tuple of (answer text, performance metrics)
        """
        result = self.invoke(query)
        return result.get("result", ""), result.get("performance", {})
