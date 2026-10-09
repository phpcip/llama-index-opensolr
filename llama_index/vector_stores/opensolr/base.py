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

import base64
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

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

from llama_index.vector_stores.opensolr.client import (
    OpensolrClient,
    OpensolrError,
    add_filters,
    apply_fresh_bias,
    apply_search_operators,
    date_range_filter,
    page_info,
    parse_operators,
)

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




_RESERVED_META = ("rtf", "uri", "url")


def _build_ingest_doc(index: str, text: str, metadata: dict, doc_id: str) -> tuple:
    """Build one Data Ingestion API document. Returns (doc, solr_id)."""
    import hashlib
    from urllib.parse import quote

    meta = dict(metadata or {})
    uri = meta.get("uri") or meta.get("url")
    if not (isinstance(uri, str) and uri.startswith(("http://", "https://"))):
        uri = f"https://ingest.opensolr.com/{index}/{quote(str(doc_id), safe='')}"
    uri = uri.rstrip("/")
    text = text or " "
    doc = {
        "uri": uri,
        "title": str(meta.get("title") or text[:100] or uri)[:250],
        "description": str(meta.get("description") or text[:200]),
        "text": text,
        "meta_ext_id": str(doc_id),
        "meta_lc_json": json.dumps(meta, ensure_ascii=False),
    }
    if meta.get("rtf"):
        doc["rtf"] = True
    if meta.get("timestamp"):
        doc["timestamp"] = meta["timestamp"]
    for key, value in meta.items():
        if isinstance(value, (str, int, float, bool)) and key not in _RESERVED_META:
            doc[_meta_field(str(key))] = str(value)
    return doc, hashlib.md5(uri.encode()).hexdigest()


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

        Runnable as-is with the public demo account: ``email="mcp@opensolr.com"``,
        ``api_key="420b8b23e7b12dc8ab838932145a5065"``, ``index_name="mcp_demo_d1__dense"``
        (300 news articles). The account is shared publicly, so other people can change
        or delete what you create, anything created there is deleted after 3 days,
        automatically, and limits are per index and small on purpose: 200 MB bandwidth,
        50 MB disk. For a private index that persists: https://opensolr.com/register
        (free forever, no card).

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
    ingest_wait: bool = True

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
        ingest_wait: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            index_name=index_name,
            create_if_missing=create_if_missing,
            location=location,
            ingest_wait=ingest_wait,
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
        """Queue nodes through the Opensolr Data Ingestion API (async).

        Embeddings and all derived fields are computed server-side; documents
        become searchable within ~1 minute. With the default
        ``ingest_wait=True`` this blocks until the queue job completes, so
        ``VectorStoreIndex.from_documents(...)`` is immediately queryable.
        """
        if not nodes:
            return []
        self._ensure_index()

        docs = []
        for node in nodes:
            meta = dict(node.metadata or {})
            if node.ref_doc_id:
                meta["ref_doc_id"] = node.ref_doc_id
            text = node.get_content(metadata_mode="none") or " "
            doc, _sid = _build_ingest_doc(self.index_name, text, meta, node.node_id)
            docs.append(doc)

        for i in range(0, len(docs), 50):
            self._client.ingest(self.index_name, docs[i : i + 50], wait=self.ingest_wait)
        return [n.node_id for n in nodes]

    def add_files(
        self,
        path: Union[str, Path],
        pattern: str = "*",
        recursive: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
        wait: Optional[bool] = None,
        timeout: float = 600.0,
    ) -> Dict[str, Any]:
        """Index local files: one file, or every file of a folder.

        Each file is read on Opensolr (its text, its pictures through OCR) and a PDF is
        indexed one document per page, so a query finds the page that answers it. Word,
        RTF, OpenDocument, text, HTML and picture files (scans) are one document each.
        ``pattern`` picks a folder's files (``"*.pdf"``), ``metadata`` goes on every
        document (filterable like any metadata). Sending the same files again replaces
        them. ``wait`` defaults to the store's ``ingest_wait``.

        Returns ``{"documents", "pending", "failed", "jobs"}``: files still being read
        when ``timeout`` ran out are under ``pending`` — call again to index them, the
        bytes are not sent twice.
        """
        self._ensure_index()
        return self._client.ingest_files(
            self.index_name, path, pattern=pattern, recursive=recursive, metadata=metadata,
            wait=self.ingest_wait if wait is None else wait, timeout=timeout,
        )

    def check_schema(self) -> Dict[str, Any]:
        """Whether this index is ready for every feature of the store, and exactly why not.

        Returns the platform's check: ``ready``, ``capabilities`` (``ingestion``,
        ``vector_search``, ``pdf_pages``, ``keyword_search``), the server, its Solr
        version, the fields and types that differ from the Opensolr vector schema,
        ``problems`` and ``fix``. The store runs this check once on its own.
        """
        return self._client.check_schema(self.index_name)

    def apply_schema(self, force: bool = False) -> Dict[str, Any]:
        """Install the Opensolr vector schema on this index (an index of a vector region).

        It replaces the index configuration; an index that already holds documents needs
        ``force=True``, and those documents must be sent again. Returns the new check.
        """
        return self._client.apply_schema(self.index_name, force=force)

    def image_to_words(self, image: Union[str, Path, bytes], top_k: int = 8) -> Dict[str, Any]:
        """Read an image and return what it was turned into, without searching yet.

        Returns ``{"text", "mode", "labels", "codes"}``: the best words to search with,
        whether it was read as visual labels ("clip") or OCR text ("ocr"), the visual
        labels (present even in OCR mode), and any barcodes / QR codes. ``image`` is a
        file path (str/Path) or raw image bytes.
        """
        raw = bytes(image) if isinstance(image, (bytes, bytearray)) else Path(image).expanduser().read_bytes()
        ans = self._client.image_to_text(self.index_name, base64.b64encode(raw).decode("ascii"), top_k=top_k)
        labels = [l["label"] for l in (ans.get("labels") or []) if isinstance(l, dict) and l.get("label")]
        codes = []
        for c in ans.get("codes") or []:
            value = c.get("text") if isinstance(c, dict) else (c if isinstance(c, str) else None)
            if value:
                codes.append(value)
        return {"text": (ans.get("text") or "").strip(), "mode": ans.get("mode") or "clip", "labels": labels, "codes": codes}

    def search_by_image(
        self,
        image: Union[str, Path, bytes],
        similarity_top_k: int = 4,
        using: str = "auto",
        mode: VectorStoreQueryMode = VectorStoreQueryMode.HYBRID,
        filters: Optional[MetadataFilters] = None,
        image_top_k: int = 8,
        **query_kwargs: Any,
    ) -> VectorStoreQueryResult:
        """Search the index with a photo instead of a text query.

        The image is turned into words by :meth:`image_to_words`, then those words run
        through :meth:`query` (``mode`` hybrid by default), so ``fresh_bias``,
        ``date_from``, ``date_to`` and ``group_pages`` apply via ``query_kwargs``. No
        image vector is stored.

        ``using``: which reading to search with — ``"auto"`` (the engine's chosen text),
        ``"meaning"`` (the visual labels), ``"text"`` (OCR text only), ``"code"`` (the
        first barcode / QR code, as an exact keyword) or ``"all"`` (labels + OCR text +
        barcodes combined). An image that gives no words returns an empty result.
        """
        read = self.image_to_words(image, top_k=image_top_k)
        if using == "meaning":
            words = ", ".join(read["labels"])
        elif using == "text":
            words = read["text"] if read["mode"] == "ocr" else ""
        elif using == "code":
            words = read["codes"][0] if read["codes"] else ""
            mode = VectorStoreQueryMode.TEXT_SEARCH
        elif using == "all":
            parts = list(read["labels"])
            if read["mode"] == "ocr" and read["text"]:
                parts.append(read["text"])
            parts.extend(read["codes"])
            words = ", ".join(p for p in parts if p)
        elif using == "auto":
            words = read["text"]
        else:
            raise ValueError("using must be one of 'auto', 'meaning', 'text', 'code', 'all'")
        if not words:
            return VectorStoreQueryResult(nodes=[], similarities=[], ids=[])
        return self.query(
            VectorStoreQuery(query_str=words, similarity_top_k=similarity_top_k, mode=mode, filters=filters),
            **query_kwargs,
        )

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
            joined = " OR ".join(f'"{_escape(i)}"' for i in node_ids)
            self._client.solr_update(
                self.index_name,
                {"delete": {"query": f"id:({joined}) OR meta_ext_id:({joined})"}},
            )

    def clear(self) -> None:
        self._ensure_index()
        self._client.solr_update(self.index_name, {"delete": {"query": "*:*"}})

    def ai_answer(
        self,
        query: str,
        filters: Optional[MetadataFilters] = None,
        # 4 documents is the platform's measured context size (OpensolrClient.RAG_DOCS);
        # this used to pass 3, which quietly overrode the client default with a smaller one.
        rag_docs: int = 4,
        rag_words: int = 1500,
        instruction: Optional[str] = None,
        tuning: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> str:
        """Grounded RAG answer generated only from this index's content.

        Two-step pattern: hybrid (BM25 + kNN) retrieval picks the top
        ``rag_docs`` hits (first ``rag_words`` words of text each), whose
        title/description/text become the LLM context — the same pipeline as
        Opensolr's hosted search UI. Pass ``instruction`` to fully control
        the prompt (e.g. "Answer in German, cite the sources you used").
        Retrieval uses the platform's tuned pipeline: your index's saved
        Search Tuning (Control Panel) applies automatically; ``tuning``
        overrides any knob per call. The list below is the whole set, not a
        sample: an abbreviated one reads as everything that is supported, and
        ``freshness_boost`` was invisible to callers because of it.
        ``fw_title``, ``fw_description``, ``fw_uri``, ``fw_text``,
        ``fw_text_t``, ``lexical_weight``, ``vector_weight``, ``vector_topk``,
        ``search_mode`` (union / keywords_required / meaning_required /
        intersection), ``quality_boost``, ``min_score``, ``freshness_boost``,
        ``fresh_bias``, ``lexical_norm_k``, ``mm`` (flexible / balanced /
        strict or raw Solr mm syntax). ``freshness_boost`` and ``fresh_bias``
        are different knobs despite the names: the first is a hard window in
        DAYS that filters older documents out, the second only re-orders,
        multiplying each score by a recency curve on ``creation_date`` so
        recent documents win ties while nothing becomes unreachable.
        Returns plain text.
        """
        fqs = _filters_to_fq(filters)
        fq = " AND ".join(f"({f})" for f in fqs) if fqs else None
        return self._client.ai_summary(
            self.index_name, query, filter_query=fq,
            rag_docs=rag_docs, rag_words=rag_words, instruction=instruction,
            tuning=tuning,
            **kwargs,
        )

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        """Vector or hybrid query. HYBRID mode uses Opensolr's ``{!hybrid}``
        parser with ``query.alpha`` as the semantic↔lexical balance.

        ``fresh_bias=True`` biases the ranking toward recent documents,
        multiplying each score by a recency curve on ``creation_date``. It
        re-orders and never filters — the hit count is unchanged, nothing old
        becomes unreachable, and a document with no ``creation_date`` is simply
        left unboosted. Works in every mode below. Off by default.

        ``date_from`` / ``date_to`` (``YYYY-MM-DD``, both included) keep only documents
        dated in that period (``creation_date``). ``group_pages`` (on by default) returns
        a PDF indexed page by page once, as its best page: the node's metadata carries
        ``page``, ``page_url`` (opens the PDF at that page), ``document_id`` and
        ``more_pages`` (its other matching pages) and its ``uri``."""
        self._ensure_index()
        k = query.similarity_top_k
        # fresh_bias arrives through **kwargs because VectorStoreQuery is
        # LlamaIndex's own dataclass and has no field for it — a keyword on the
        # call is the only place a caller can put it without a fork of the schema.
        fresh_bias = bool(kwargs.get("fresh_bias", False))
        group_pages = bool(kwargs.get("group_pages", True))
        # Checked before anything is embedded: a malformed date is the caller's, not a GPU call's
        period = date_range_filter(kwargs.get("date_from"), kwargs.get("date_to"))

        params: Dict[str, Any] = {"rows": k, "fl": "*,score"}

        # Search operators (+word, -word, +"phrase", -"phrase") come out of the text once,
        # for every mode below. See parse_operators() for why they cannot stay inside the
        # query on any path that involves a vector.
        ops = parse_operators(query.query_str or "")
        # Nothing left once the operators are removed ("-Ruben" alone): the operators ARE the
        # query. Hand the whole string to edismax, which understands them natively, and emit
        # no filters.
        ops_only = ops["has_ops"] and len(ops["base"]) < 2
        lexical_text = (query.query_str or "") if (not ops["has_ops"] or ops_only) else ops["base"]

        if query.mode == VectorStoreQueryMode.TEXT_SEARCH or ops_only:
            # Pure lexical (edismax): no embedding call, zero AI quota,
            # works on any Opensolr index including non-vector ones.
            if not query.query_str:
                raise ValueError("TEXT_SEARCH mode needs query_str")
            # Bound by reference (2026-09-09) instead of inlined: inlining meant a '}' in the
            # caller's text closed the local-param block, which is why braces AND quotes were
            # stripped first — and stripping quotes silently broke every phrase query.
            params["uq"] = lexical_text
            params["q"] = '{!edismax qf="title^100 description^20 text^1" v=$uq}'
            knn = None
        else:
            # An index without the vector fields says so before an embedding is paid for
            self._client.require_capability(self.index_name, "vector_search")
            vector = query.query_embedding
            if vector is None:
                if not query.query_str:
                    raise ValueError("Query needs query_embedding or query_str")
                # The embedder must never see an operator — it has no concept of negation, so
                # a '-' term reads as one more word and pulls results towards what the caller
                # asked to remove.
                vector = self._client.embed(
                    self.index_name,
                    ops["base"] if ops["has_ops"] else query.query_str,
                    is_query=True,
                )
            compact = json.dumps(vector, separators=(",", ":"))
            knn = f"{{!knn f=embeddings topK={max(k, 10)}}}{compact}"
        if knn is None:
            pass
        elif query.mode in (VectorStoreQueryMode.HYBRID, VectorStoreQueryMode.SEMANTIC_HYBRID) and query.query_str:
            alpha = query.alpha if query.alpha is not None else 0.5
            params["uq"] = lexical_text
            params["q"] = (
                f"{{!hybrid lexical=$lexicalRaw vector=$vectorQuery "
                f"mode=union alpha={alpha} topN={max(k, 10)}}}"
            )
            params["lexicalRaw"] = '{!edismax qf="title^100 text^1" v=$uq}'
            params["vectorQuery"] = knn
        elif knn is not None:
            params["q"] = knn
        # Operators become filters on both vector-bearing shapes. On the pure-kNN shape this
        # is the only thing that can honour them at all — that query has no edismax in it.
        if knn is not None:
            apply_search_operators(params, ops)

        # Fresh Results Bias wraps whichever shape was built above — edismax,
        # fused {!hybrid} or bare {!knn} — so the recency multiplier reaches every
        # candidate, including the vector-only ones an edismax bf never sees.
        if fresh_bias:
            apply_fresh_bias(params)

        fq = _filters_to_fq(query.filters)
        if query.node_ids:
            joined = " OR ".join(f'"{_escape(i)}"' for i in query.node_ids)
            fq.append(f"id:({joined})")
        if query.doc_ids:
            joined = " OR ".join(f'"{_escape(i)}"' for i in query.doc_ids)
            fq.append(f'{_meta_field("ref_doc_id")}:({joined})')
        # EXTEND, never assign: the search-operator filters and Fresh Results Bias's
        # "must have a creation_date" clause are already in params by now.
        add_filters(params, *fq, period)

        body = self._client.select_grouped(self.index_name, params, group_pages)
        expanded = body.get("expanded") or {}

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
            ext = _flat(doc.get("meta_ext_id"))
            node_id = str(ext) if ext else str(_flat(doc.get("id", "")))
            # A page of a PDF also says where it lives (uri, page, page_url, document_id, more_pages);
            # any other node keeps exactly the metadata it was sent with
            where = page_info(doc, expanded)
            if where.get("page"):
                for key, value in where.items():
                    if value not in ("", None):
                        metadata.setdefault(key, value)
                # every page of a PDF carries the PDF's meta_ext_id: the page number keeps node ids apart
                node_id = f"{node_id}#page={where['page']}"
            nodes.append(TextNode(id_=node_id, text=str(text), metadata=metadata))
            similarities.append(float(_flat(doc.get("score", 0.0)) or 0.0))
            ids.append(node_id)

        return VectorStoreQueryResult(nodes=nodes, similarities=similarities, ids=ids)
