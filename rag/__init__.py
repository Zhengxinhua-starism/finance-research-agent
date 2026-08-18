"""Agentic RAG 知识库层。

检索链路：query → [BM25 关键词路 ‖ Chroma 向量路] → RRF 融合 → CrossEncoder 精排 → top_k
两路召回互补、融合用名次而非分数、精排换取精度，三个环节各自解决一个具体问题。
"""

from rag.chunker import TextChunker
from rag.hybrid_retriever import BM25Index, HybridRetriever, reciprocal_rank_fusion, tokenize
from rag.rag_store import RagStore, embed_texts, get_embedding_model
from rag.reranker import CrossEncoderReranker

__all__ = [
    "BM25Index",
    "CrossEncoderReranker",
    "HybridRetriever",
    "RagStore",
    "TextChunker",
    "embed_texts",
    "get_embedding_model",
    "reciprocal_rank_fusion",
    "tokenize",
]
