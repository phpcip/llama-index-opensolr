"""Opensolr vector store for LlamaIndex — managed Apache Solr 9.x with
server-side GPU embeddings and native hybrid (BM25 + kNN) search."""

from llama_index.vector_stores.opensolr.base import OpensolrVectorStore
from llama_index.vector_stores.opensolr.client import OpensolrClient, OpensolrError

__all__ = ["OpensolrVectorStore", "OpensolrClient", "OpensolrError"]
