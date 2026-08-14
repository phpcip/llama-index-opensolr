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

MIT license.
