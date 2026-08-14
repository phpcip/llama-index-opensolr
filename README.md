# llama-index-opensolr

LlamaIndex integration for [Opensolr](https://opensolr.com) — managed Apache
Solr as a vector store, with **server-side embeddings** and native **hybrid
(BM25 + kNN) search**.

No local embedding model. No third-party embedding API key. One set of
credentials; vectors are computed on Opensolr's GPU infrastructure
(multilingual E5-large-instruct, 1024 dimensions, cosine).

**Product page:** [opensolr.com/langchain](https://opensolr.com/langchain) ·
free 15-day trial, no card, at [opensolr.com](https://opensolr.com)

```bash
pip install llama-index-opensolr
```

## Quickstart

```python
from llama_index.core import VectorStoreIndex, StorageContext, Document
from llama_index.vector_stores.opensolr import OpensolrVectorStore
from llama_index.embeddings.opensolr import OpensolrEmbedding

store = OpensolrVectorStore(
    index_name="mysite__dense",
    email="you@example.com",
    api_key="YOUR_OPENSOLR_API_KEY",
    create_if_missing=True,
)
embed_model = OpensolrEmbedding(
    email="you@example.com", api_key="YOUR_OPENSOLR_API_KEY",
    index_name="mysite__dense",
)

index = VectorStoreIndex.from_documents(
    [Document(text="Hybrid search fuses BM25 with vector similarity")],
    storage_context=StorageContext.from_defaults(vector_store=store),
    embed_model=embed_model,
)

retriever = index.as_retriever(similarity_top_k=5)
print(retriever.retrieve("how do keyword and semantic search combine?"))
```

## Hybrid search

Opensolr fuses BM25 and kNN scores **per document** with its native
`{!hybrid}` Solr query parser:

```python
from llama_index.core.vector_stores.types import VectorStoreQuery, VectorStoreQueryMode

result = store.query(VectorStoreQuery(
    query_str="affordable restaurants",
    similarity_top_k=5,
    mode=VectorStoreQueryMode.HYBRID,
    alpha=0.5,          # 0 = all semantic … 1 = all lexical
))
```

## Metadata filters

Standard LlamaIndex `MetadataFilters` (EQ, NE, IN, NIN, GT/GTE, LT/LTE) map
to Solr `fq` — and every index is also plain Apache Solr with the native
`/select` API when you need facets, highlighting, or anything beyond retrieval.

## Notes

- Vector-enabled indexes run on Opensolr's Solr 9.x environments — currently
  `us` (Chicago), `de` (Germany), `fi` (Finland). The list is fetched live
  from the platform; **additional dedicated regions can be deployed on
  request** (paid add-on): [support@opensolr.com](mailto:support@opensolr.com).
- Siblings: [`langchain-opensolr`](https://pypi.org/project/langchain-opensolr/)
  (LangChain) · [`opensolr-mcp`](https://pypi.org/project/opensolr-mcp/)
  (MCP server for agents).

## How writing works (Data Ingestion API)

Writes go through Opensolr's [Data Ingestion API](https://opensolr.com/learn/api-data-ingestion/204/data-ingestion-api-push-documents-to-your-opensolr-index-programmatically)
— the same pipeline the Drupal and WordPress connectors use. It is
**asynchronous**: documents are queued, then embeddings, sentiment, language
and all crawler-identical derived fields are computed **server-side**, and
documents become searchable within about a minute. Progress is visible in the
Opensolr Control Panel and via the `ingest_status` API. Each document's
identity is its `uri` (the Solr id is `md5(uri)`): pass a real URL in
metadata (`{"uri": "https://..."}`), or a deterministic one is synthesized
from your id. Re-submitting the same `uri` updates the document. Pass
`{"rtf": True, "uri": "https://.../file.pdf"}` and the server extracts the
text from PDF/DOCX/XLSX for you.

## Lexical-only mode

Don't need vectors? Pure keyword search skips the embedding call entirely —
zero AI quota, and it works on **any** Opensolr index, including non-vector
ones and older Solr versions.

## Your index schema

Documents follow the Opensolr document model (`title`, `description`, `text`,
`meta_*` custom fields). To see the full schema: **Control Panel → click your
index → Configuration → Edit File → schema.xml**. Prefer zero-effort data
entry? Configure the **Web Crawler** in the Control Panel (Index Tools →
WebCrawler): add your site URL, validate it, and Opensolr indexes the whole
site for you.

MIT license.
