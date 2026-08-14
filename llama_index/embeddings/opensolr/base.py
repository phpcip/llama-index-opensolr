"""Opensolr embedding model for LlamaIndex — no local model, no third-party
API key: texts are embedded server-side on Opensolr's GPU infrastructure."""

from __future__ import annotations

from typing import Any, List, Optional

from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.bridge.pydantic import PrivateAttr

from llama_index.vector_stores.opensolr.client import OpensolrClient


class OpensolrEmbedding(BaseEmbedding):
    """Embeddings backed by the Opensolr AI API (1024-dim, cosine).

    Example:
        .. code-block:: python

            from llama_index.embeddings.opensolr import OpensolrEmbedding

            embed_model = OpensolrEmbedding(
                email="you@example.com",
                api_key="...",
                index_name="mysite__dense",
            )

    Args:
        email: Opensolr account email.
        api_key: Opensolr API key.
        index_name: Vector-enabled index the embedding usage is accounted to.
    """

    index_name: str = ""

    _client: OpensolrClient = PrivateAttr()

    def __init__(
        self,
        email: str = "",
        api_key: str = "",
        index_name: str = "",
        client: Optional[OpensolrClient] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(index_name=index_name, model_name="opensolr-e5-large-instruct", **kwargs)
        if client is None:
            if not (email and api_key):
                raise ValueError("Provide either a client or email + api_key")
            client = OpensolrClient(email, api_key)
        if not index_name:
            raise ValueError("index_name is required (usage is accounted per index)")
        self._client = client

    @classmethod
    def class_name(cls) -> str:
        return "OpensolrEmbedding"

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._client.embed(self.index_name, query, is_query=True)

    def _get_text_embedding(self, text: str) -> List[float]:
        return self._client.batch_embed(self.index_name, [text])[0]

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return self._client.batch_embed(self.index_name, texts)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)
