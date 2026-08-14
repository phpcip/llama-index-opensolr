"""Opensolr vector store implementation for LlamaIndex.

Backed by a managed, vector-enabled Opensolr index (Apache Solr 9.x,
``knn_vector`` 1024-dim, cosine). Embeddings are computed **server-side** on
Opensolr's GPU infrastructure, so this store works without any local
embed model — pair it with
:class:`llama_index.embeddings.opensolr.OpensolrEmbedding`, or let the store
embed transparently when nodes/queries arrive without embeddings.

Supports ``VectorStoreQueryMode.HYBRID`` natively via Opensolr's ``{!hybrid}``
Solr query parser (per-document BM25 + kNN score fusion, ``alpha`` balance).
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.schema import BaseNode, TextNode
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    FilterOperator,
    MetadataFilters,
    VectorStoreQuery,
    VectorStoreQueryMode,
    VectorStoreQueryResult,
)

from llama_index.vector_stores.opensolr.client import OpensolrClient, OpensolrError

_META_KEY_RE = re.compile(r"[^a-z0-9_]+")


def _meta_field(key: str) -> str:
    return f"meta_{_META_KEY_RE.sub('_', key.lower()).strip('_')}"


def _escape(value: Any) -> str:
    return str(value).replace("\\", "\\\\").replace('"', '\\"')


def _filters_to_fq(filters: Optional[MetadataFilters]) -> List[str]:
    if filters is None:
        return []
    fq: List[str] = []
    for f in filters.filters:
        if isinstance(f, MetadataFilters):
            fq.extend(_filters_to_fq(f))
            continue
        field = _meta_field(f.key)
        op = f.operator
        if op == FilterOperator.EQ:
            fq.append(f'{field}:"{_escape(f.value)}"')
        elif op == FilterOperator.NE:
            fq.append(f'-{field}:"{_escape(f.value)}"')
        elif op == FilterOperator.IN:
            joined = " OR ".join(f'"{_escape(v)}"' for v in f.value)
            fq.append(f"{field}:({joined})")
        elif op == FilterOperator.NIN:
            joined = " OR ".join(f'"{_escape(v)}"' for v in f.value)
            fq.append(f"-{field}:({joined})")
        elif op in (FilterOperator.GT, FilterOperator.GTE):
            lo = f'"{_escape(f.value)}"'
            bracket = "{" if op == FilterOperator.GT else "["
            fq.append(f"{field}:{bracket}{lo} TO *]")
        elif op in (FilterOperator.LT, FilterOperator.LTE):
            hi = f'"{_escape(f.value)}"'
            bracket = "}" if op == FilterOperator.LT else "]"
            fq.append(f"{field}:[* TO {hi}{bracket}")
        else:
            raise ValueError(f"Unsupported filter operator: {op}")
    return fq


class OpensolrVectorStore(BasePydanticVectorStore):
    """Managed hybrid vector store on Opensolr.

    Example:
        .. code-block:: python

            from llama_index.vector_stores.opensolr import OpensolrVectorStore
            from llama_index.core import VectorStoreIndex, StorageContext

            store = OpensolrVectorStore(
                index_name="mysite__dense",
                email="you@example.com",
                api_key="YOUR_OPENSOLR_API_KEY",
            )
            storage = StorageContext.from_defaults(vector_store=store)
            index = VectorStoreIndex.from_documents(docs, storage_context=storage)

    Args:
        index_name: Vector-enabled Opensolr index (Solr 9.x locations:
            us, de, fi — more on request).
        email: Opensolr account email.
        api_key: Opensolr API key.
        create_if_missing: Provision the index automatically on first use.
        location: Where to create it (``us``, ``de``, ``fi``).
    """

    stores_text: bool = True

    index_name: str
    create_if_missing: bool = False
    location: str = "us"

    _client: OpensolrClient = PrivateAttr()
    _checked: bool = PrivateAttr(default=False)

    def __init__(
        self,
        index_name: str,
        email: str = "",
        api_key: str = "",
        client: Optional[OpensolrClient] = None,
        create_if_missing: bool = False,
        location: str = "us",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            index_name=index_name,
            create_if_missing=create_if_missing,
            location=location,
            **kwargs,
        )
        if client is None:
            if not (email and api_key):
                raise ValueError("Provide either a client or email + api_key")
            client = OpensolrClient(email, api_key)
        self._client = client

    @classmethod
    def class_name(cls) -> str:
        return "OpensolrVectorStore"

    @property
    def client(self) -> OpensolrClient:
        return self._client

    # ------------------------------------------------------------------ #

    def _ensure_index(self) -> None:
        if self._checked:
            return
        try:
            self._client.get_core_info(self.index_name)
        except OpensolrError:
            if not self.create_if_missing:
                raise
            self._client.create_index(self.index_name, self.location)
            import time

            for _ in range(5):
                time.sleep(2)
                try:
                    self._client.get_core_info(self.index_name, refresh=True)
                    break
                except OpensolrError:
                    continue
        self._checked = True

    def add(self, nodes: List[BaseNode], **kwargs: Any) -> List[str]:
        """Index nodes. Nodes without embeddings are embedded server-side."""
        if not nodes:
            return []
        self._ensure_index()

        texts = [n.get_content(metadata_mode="none") or " " for n in nodes]
        embeddings: List[Optional[List[float]]] = [n.get_embedding() if n.embedding is not None else None for n in nodes]
        missing = [i for i, e in enumerate(embeddings) if e is None]
        if missing:
            computed = self._client.batch_embed(self.index_name, [texts[i] for i in missing])
            for i, vec in zip(missing, computed):
                embeddings[i] = vec

        docs = []
        for node, text, vector in zip(nodes, texts, embeddings):
            meta = dict(node.metadata or {})
            doc: Dict[str, Any] = {
                "id": node.node_id,
                "text": text,
                "embeddings": vector,
                "meta_lc_json": json.dumps(meta, ensure_ascii=False),
                "title": str(meta.get("title") or text[:100]),
            }
            if node.ref_doc_id:
                doc[_meta_field("ref_doc_id")] = node.ref_doc_id
            for key, value in meta.items():
                if isinstance(value, (str, int, float, bool)):
                    doc[_meta_field(str(key))] = str(value)
            docs.append(doc)

        self._client.solr_update(self.index_name, docs)
        return [n.node_id for n in nodes]

    def delete(self, ref_doc_id: str, **kwargs: Any) -> None:
        """Delete all nodes belonging to a source document."""
        self._ensure_index()
        self._client.solr_update(
            self.index_name,
            {"delete": {"query": f'{_meta_field("ref_doc_id")}:"{_escape(ref_doc_id)}"'}},
        )

    def delete_nodes(self, node_ids: Optional[List[str]] = None, **kwargs: Any) -> None:
        if node_ids:
            self._ensure_index()
            self._client.solr_update(self.index_name, {"delete": list(node_ids)})

    def clear(self) -> None:
        self._ensure_index()
        self._client.solr_update(self.index_name, {"delete": {"query": "*:*"}})

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        """Vector or hybrid query. HYBRID mode uses Opensolr's ``{!hybrid}``
        parser with ``query.alpha`` as the semantic↔lexical balance."""
        self._ensure_index()
        k = query.similarity_top_k

        vector = query.query_embedding
        if vector is None:
            if not query.query_str:
                raise ValueError("Query needs query_embedding or query_str")
            vector = self._client.embed(self.index_name, query.query_str, is_query=True)

        compact = json.dumps(vector, separators=(",", ":"))
        knn = f"{{!knn f=embeddings topK={max(k, 10)}}}{compact}"

        params: Dict[str, Any] = {"rows": k, "fl": "*,score"}
        if query.mode in (VectorStoreQueryMode.HYBRID, VectorStoreQueryMode.SEMANTIC_HYBRID) and query.query_str:
            alpha = query.alpha if query.alpha is not None else 0.5
            clean = query.query_str.replace("{", " ").replace("}", " ").replace('"', " ")
            params["q"] = (
                f"{{!hybrid lexical=$lexicalRaw vector=$vectorQuery "
                f"mode=union alpha={alpha} topN={max(k, 10)}}}"
            )
            params["lexicalRaw"] = f'{{!edismax qf="title^100 text^1"}}{clean}'
            params["vectorQuery"] = knn
        else:
            params["q"] = knn

        fq = _filters_to_fq(query.filters)
        if query.node_ids:
            joined = " OR ".join(f'"{_escape(i)}"' for i in query.node_ids)
            fq.append(f"id:({joined})")
        if query.doc_ids:
            joined = " OR ".join(f'"{_escape(i)}"' for i in query.doc_ids)
            fq.append(f'{_meta_field("ref_doc_id")}:({joined})')
        if fq:
            params["fq"] = fq

        body = self._client.solr_select(self.index_name, params)

        nodes: List[TextNode] = []
        similarities: List[float] = []
        ids: List[str] = []
        for doc in body["response"]["docs"]:
            def _flat(v: Any) -> Any:
                return v[0] if isinstance(v, list) and len(v) == 1 else v

            metadata: Dict[str, Any] = {}
            raw = _flat(doc.get("meta_lc_json"))
            if raw:
                try:
                    metadata = json.loads(raw)
                except (TypeError, json.JSONDecodeError):
                    metadata = {}
            text = _flat(doc.get("text", "")) or ""
            if isinstance(text, list):
                text = " ".join(str(t) for t in text)
            node_id = str(_flat(doc.get("id", "")))
            nodes.append(TextNode(id_=node_id, text=str(text), metadata=metadata))
            similarities.append(float(_flat(doc.get("score", 0.0)) or 0.0))
            ids.append(node_id)

        return VectorStoreQueryResult(nodes=nodes, similarities=similarities, ids=ids)
