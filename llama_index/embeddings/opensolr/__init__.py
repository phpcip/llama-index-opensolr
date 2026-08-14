"""Opensolr server-side embeddings for LlamaIndex (E5-large-instruct,
multilingual, 1024 dimensions, computed on Opensolr's GPU infrastructure)."""

from llama_index.embeddings.opensolr.base import OpensolrEmbedding

__all__ = ["OpensolrEmbedding"]
