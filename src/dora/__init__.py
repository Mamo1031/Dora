"""Dora: Knowledge-augmented local LLM using RAG."""

from dora.knowledge_base import KnowledgeBase
from dora.llm import LocalLLM
from dora.rag import RAGChain

__version__ = "0.0.1"
__all__ = ["KnowledgeBase", "LocalLLM", "RAGChain"]
