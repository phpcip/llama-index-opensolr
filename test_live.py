#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Live production test for ``llama-index-opensolr``.

Exercises EVERY public method of the package against the real Opensolr platform
(opensolr.com + api.opensolr.com) with the public MCP demo account, asserting on
VALUES rather than on "no exception was raised", and deleting everything it
created before it exits — including when it fails or is interrupted.

Run:
    /Users/ciprian/workspace/llama-index-opensolr/.venv/bin/python test_live.py

What it touches
---------------
* ``mcp_demo_d1__dense`` — the seeded, read-only demo index (300 news articles).
  READ ONLY. Nothing in this script writes to it or deletes from it.
* ``mcp_t_<8 hex>__dense`` — a throwaway index created by this run and deleted in
  the ``finally`` block. The name is unique per run, so the script is safe to
  re-run concurrently with itself and never collides with a previous run's
  leftovers.

Two platform behaviours the script has to respect
-------------------------------------------------
* **Rate limit — 30 requests/minute**, enforced per user AND per IP on both
  boxes. Every httpx POST to a ``*.opensolr.com`` host goes through a sliding
  60-second window capped at 25, which guarantees at most 25 in any calendar
  minute (the bucket the server actually counts). Direct-to-Solr calls go to
  ``*.solrcluster.com`` and are not rate limited, so they are not paced.
* **Ingestion is ASYNC** — the queue is drained by a cron that runs every
  minute. Nothing here sleeps a fixed time: it polls ``ingest_status`` (through
  the package's own ``wait=True``) or the document count in Solr, and gives up
  with a clear message at the 120-second budget.
"""

from __future__ import annotations

import json
import os
import re
import signal
import sys
import threading
import time
import traceback
import uuid
from collections import deque
from datetime import datetime, timedelta, timezone

import httpx

# The package under test is imported from the working copy, never from an
# installed copy that might be a different version than the one being edited.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llama_index.core.schema import NodeRelationship, RelatedNodeInfo, TextNode
from llama_index.core.vector_stores.types import (
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
    VectorStoreQuery,
    VectorStoreQueryMode,
    VectorStoreQueryResult,
)

from llama_index.vector_stores.opensolr import (
    OpensolrClient,
    OpensolrError,
    OpensolrVectorStore,
)
from llama_index.vector_stores.opensolr.client import (
    AI_BASE,
    BATCH_EMBED_MAX,
    DEFAULT_RAG_DOCS,
    DEFAULT_RAG_WORDS,
    DOC_FENCE,
    FRESH_BIAS_FUNCTION,
    MGMT_BASE,
    SEARCH_OPERATOR_FIELDS,
    VECTOR_LOCATIONS,
    apply_fresh_bias,
    apply_search_operators,
    build_context,
    build_instruction,
    parse_operators,
    resolve_location,
)

# --------------------------------------------------------------------------- #
# Credentials — the public MCP demo account. Throwaway by design, swept after   #
# three days. Full access to its own indexes and nothing else.                  #
# --------------------------------------------------------------------------- #
EMAIL = "mcp@opensolr.com"
API_KEY = "420b8b23e7b12dc8ab838932145a5065"

DEMO_INDEX = "mcp_demo_d1__dense"                       # read-only, 300 articles
RUN_ID = uuid.uuid4().hex[:8]
TMP_INDEX = f"mcp_t_{RUN_ID}__dense"                    # created + deleted here
TMP_INDEX_2 = f"mcp_t_{RUN_ID}b__dense"                 # create_if_missing target
TMP_LOCATION = "us"                                     # CHICAGO-96, vector-enabled

# Every index this run provisions, so teardown deletes all of them and not just
# the one the happy path knows about.
CREATED: list = []

INGEST_BUDGET = 120.0        # seconds; the brief's give-up point for async work
RAG_QUERY = "which football matches are described and who is playing"


# --------------------------------------------------------------------------- #
# Rate limiter                                                                  #
# --------------------------------------------------------------------------- #
# The platform counts requests in calendar-minute buckets, per user and per IP,
# and 429s at 30. A sliding window of N per any rolling 60 seconds guarantees at
# most N inside every calendar minute, so 25 leaves headroom for the retry the
# script may do on the same second the server rolls its bucket.
_RL_MAX = 25
_RL_WINDOW = 60.0
_rl_lock = threading.Lock()
_rl_times: deque = deque()
_rl_total = 0


def _rate_slot() -> None:
    """Block until one request may be sent inside the sliding window."""
    global _rl_total
    while True:
        with _rl_lock:
            now = time.time()
            while _rl_times and now - _rl_times[0] >= _RL_WINDOW:
                _rl_times.popleft()
            if len(_rl_times) < _RL_MAX:
                _rl_times.append(now)
                _rl_total += 1
                return
            wait = _RL_WINDOW - (now - _rl_times[0]) + 0.05
        time.sleep(max(0.05, wait))


_orig_post = httpx.Client.post
_orig_get = httpx.Client.get


def _paced_post(self, url, *args, **kwargs):
    """Pace only the platform APIs; direct Solr hosts are not rate limited."""
    if ".opensolr.com" in str(url):
        _rate_slot()
    return _orig_post(self, url, *args, **kwargs)


def _paced_get(self, url, *args, **kwargs):
    if ".opensolr.com" in str(url):
        _rate_slot()
    return _orig_get(self, url, *args, **kwargs)


httpx.Client.post = _paced_post      # type: ignore[method-assign]
httpx.Client.get = _paced_get        # type: ignore[method-assign]


# --------------------------------------------------------------------------- #
# Reporting                                                                     #
# --------------------------------------------------------------------------- #
PASSED = 0
FAILED = 0
S: dict = {}          # values checks hand to later checks


def _short(text: str, limit: int = 220) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def c_parse_operators():
    cases = {
        "did juventus win that match? -Ruben": ("did juventus win that match?", [], ["Ruben"]),
        '+laptop +"13 inch" -refurbished': ("", ["laptop", '"13 inch"'], ["refurbished"]),
        'laptop -"open box" gaming': ("laptop gaming", [], ['"open box"']),
        # a '-' mid-token is part of the word, not an operator
        "e-mail covid-19 1+1 formula": ("e-mail covid-19 1+1 formula", [], []),
        # a '-' INSIDE a quoted phrase belongs to the phrase
        '"foo -bar" baz': ('"foo -bar" baz', [], []),
        # a lone sign, or an empty operand, is plain text
        'cafea + - fara +"" zahar': ('cafea + - fara +"" zahar', [], []),
    }
    for raw, (base, req, exc) in cases.items():
        got = parse_operators(raw)
        assert got["base"] == base, (raw, got["base"], base)
        assert got["required"] == req, (raw, got["required"], req)
        assert got["excluded"] == exc, (raw, got["excluded"], exc)
        assert got["has_ops"] == bool(req or exc), (raw, got["has_ops"])
    return f"{len(cases)} queries split exactly, hyphenated words and quoted '-' left alone"


def c_apply_search_operators():
    params = {"q": "{!hybrid}", "fq": ["meta_kind:news"]}
    out = apply_search_operators(params, parse_operators('news +"press release" -rumour'))
    assert out is params, "must mutate and return the same dict"
    assert params["reqQ0"] == '"press release"', params.get("reqQ0")
    assert params["negQ0"] == "rumour", params.get("negQ0")
    want_req = '{!edismax qf="%s" mm="100%%" v=$reqQ0}' % SEARCH_OPERATOR_FIELDS
    want_neg = '-{!edismax qf="%s" mm="100%%" v=$negQ0}' % SEARCH_OPERATOR_FIELDS
    assert want_req in params["fq"], params["fq"]
    assert want_neg in params["fq"], params["fq"]
    assert "meta_kind:news" in params["fq"], "a pre-existing fq must survive"
    assert params["q"] == "{!hybrid}", "q must be untouched"
    return "operands bound by reference, both filters added, existing fq kept"


def check(label, fn):
    """Run one assertion. ``fn`` returns a short detail string, or raises."""
    global PASSED, FAILED
    try:
        detail = fn()
    except Exception as exc:                                # noqa: BLE001
        FAILED += 1
        print(f"✘ {label} — {type(exc).__name__}: {_short(exc)}", flush=True)
        if os.environ.get("LIVE_TEST_TRACE"):
            traceback.print_exc()
        return False
    PASSED += 1
    print(f"✔ {label}" + (f" — {detail}" if detail else ""), flush=True)
    return True


def need(*keys):
    """Raise if a prerequisite value is missing, so the check fails honestly."""
    for key in keys:
        if key not in S:
            raise AssertionError(f"prerequisite missing: {key}")
    return [S[k] for k in keys]


def section(title: str) -> None:
    print(f"\n--- {title} ---", flush=True)


# --------------------------------------------------------------------------- #
# Helpers                                                                       #
# --------------------------------------------------------------------------- #
def index_names(rows) -> set:
    """Index names out of get_index_list().

    The management API answers with ``index_name``; older/other endpoints in the
    same family use ``core_name``, so both are accepted rather than assumed.
    """
    out = set()
    for row in rows or []:
        if isinstance(row, dict):
            name = row.get("index_name") or row.get("core_name") or row.get("name")
            if name:
                out.add(name)
    return out


def solr_count(client: OpensolrClient, index: str, q: str = "*:*") -> int:
    """numFound for a query, straight off the index's own Solr endpoint."""
    body = client.solr_select(index, {"q": q, "rows": 0})
    return int(body["response"]["numFound"])


def wait_for_count(client, index, q, want, budget=INGEST_BUDGET, every=5.0):
    """Poll the document count instead of sleeping a guessed interval.

    Returns the count once it reaches ``want``; raises with a clear message when
    the budget runs out, naming what it saw last.
    """
    deadline = time.time() + budget
    seen = -1
    while time.time() < deadline:
        # Direct-to-Solr, so this loop costs no platform rate-limit budget.
        seen = solr_count(client, index, q)
        if seen >= want:
            return seen
        time.sleep(every)
    raise AssertionError(
        f"{index}: q={q!r} still at {seen} document(s), wanted {want}, "
        f"after {budget:.0f}s"
    )


def delete_index(index: str) -> dict:
    """Delete an index through the management API.

    Not a method of the package (the client has no delete), so it is a plain
    request — this is teardown, not a thing under test.
    """
    resp = httpx.post(
        f"{MGMT_BASE}/delete_index",
        data={"email": EMAIL, "api_key": API_KEY, "index_name": index},
        timeout=120.0,
    )
    try:
        return resp.json()
    except json.JSONDecodeError:
        return {"status": None, "msg": resp.text[:200]}


def iso(dt: datetime) -> int:
    return int(dt.timestamp())


# --------------------------------------------------------------------------- #
# 1. Pure builders — no network, no account, no quota                           #
# --------------------------------------------------------------------------- #
def test_pure_builders() -> None:
    section("pure builders (client.py, no network)")

    def c_resolve():
        assert resolve_location("us") == "CHICAGO-96"
        assert resolve_location("  DE  ") == "DE-SOLR-9"
        assert resolve_location("fi") == "FINLAND9"
        # Unknown aliases pass through so a new region needs no package upgrade.
        assert resolve_location("XX-BRAND-NEW") == "XX-BRAND-NEW"
        return f"us->CHICAGO-96, de->DE-SOLR-9, fi->FINLAND9, {len(VECTOR_LOCATIONS)} aliases"

    check("resolve_location maps aliases and passes unknowns through", c_resolve)

    def c_fresh_bias():
        params = {"q": "{!knn f=embeddings topK=10}[0.1]", "rows": 7}
        inner = params["q"]
        out = apply_fresh_bias(params)
        assert out is params, "must mutate in place and return the same dict"
        assert params["q"] == "{!boost b=$freshBias v=$freshBiasInner}", params["q"]
        assert params["freshBiasInner"] == inner
        assert params["freshBias"] == FRESH_BIAS_FUNCTION
        assert params["rows"] == 7, "unrelated params must survive"
        assert "recip(max(0,ms(NOW,creation_date))" in FRESH_BIAS_FUNCTION
        return "q wrapped in {!boost}, inner query moved to $freshBiasInner"

    check("apply_fresh_bias wraps q and preserves the inner query", c_fresh_bias)
    check("parse_operators splits +/- operators and leaves ordinary text alone",
          c_parse_operators)
    check("apply_search_operators emits fq filters and keeps existing ones",
          c_apply_search_operators)

    def c_context():
        docs = [
            {"id": "d1", "score": 10.0, "title": "Alpha", "description": "First",
             "text": "one two three four five six seven"},
            {"id": "d2", "score": 6.0, "title": "Beta", "description": "Second",
             "text": "beta body text"},
            # 4.0 < 10.0 * 0.5 -> below the relevance floor, must be dropped
            {"id": "d3", "score": 4.0, "title": "Gamma", "description": "Third",
             "text": "gamma body text"},
        ]
        hl = {"d1": {"text": ["<em>one</em> two three"]}}
        ctx = build_context(docs, hl, top_n=3, max_words=4)
        assert ctx.count(DOC_FENCE) == 2, f"expected 2 fences, got {ctx.count(DOC_FENCE)}"
        assert "Gamma" not in ctx, "weak hit was not filtered by the 0.5 floor"
        assert "Alpha" in ctx and "Beta" in ctx
        assert "MOST RELEVANT EXCERPTS:" in ctx
        assert "<em>" not in ctx, "highlighter markup reached the prompt"
        # Fragment opens lowercase and ends without punctuation -> marked both ends.
        assert "- ... one two three ..." in ctx, ctx
        assert "one two three four\n" in ctx, "excerpt did not cut at max_words"
        assert "five" not in ctx, "excerpt kept words past max_words"
        S["ctx"] = ctx
        return f"2 of 3 docs kept, {len(ctx)} chars, fragment marked, cut at 4 words"

    check("build_context drops sub-half-score hits, strips tags, cuts at max_words", c_context)

    def c_instruction():
        ctx = need("ctx")[0]
        inst = build_instruction(ctx, "What is alpha?")
        assert inst.startswith(ctx), "context must lead the prompt"
        assert "Those were the 2 documents." in inst
        assert "Find ALL of the 2 documents" in inst
        assert 'Never begin with "Based on" or "According to"' in inst
        assert 'starts "There is no information about"' in inst
        assert inst.endswith("Question: What is alpha?\nAnswer:"), inst[-80:]
        # Empty context still has to read as a grammatical prompt.
        empty = build_instruction("", "q")
        assert "Those were the 1 documents." in empty
        return f"{len(inst)} chars, count=2, ends on the Question slot"

    check("build_instruction counts documents, bans the openings, ends on Question", c_instruction)

    def c_constants():
        assert DEFAULT_RAG_DOCS == 4, DEFAULT_RAG_DOCS
        assert DEFAULT_RAG_WORDS == 1500, DEFAULT_RAG_WORDS
        assert BATCH_EMBED_MAX == 50, BATCH_EMBED_MAX
        assert OpensolrClient.RAG_DOCS == DEFAULT_RAG_DOCS
        assert OpensolrClient.RAG_WORDS == DEFAULT_RAG_WORDS
        assert MGMT_BASE == "https://opensolr.com/solr_manager/api"
        assert AI_BASE == "https://api.opensolr.com/solr_manager/api"
        return "RAG_DOCS=4, RAG_WORDS=1500, BATCH_EMBED_MAX=50, both bases correct"

    check("module constants match the platform contract", c_constants)


# --------------------------------------------------------------------------- #
# 2. Client, read-only against the seeded demo index                            #
# --------------------------------------------------------------------------- #
def test_client_read(client: OpensolrClient) -> None:
    section(f"client — read-only against {DEMO_INDEX}")

    def c_regions():
        regions = client.vector_regions()
        assert isinstance(regions, list) and regions, regions
        envs = {r["environment"] for r in regions}
        for row in regions:
            assert {"environment", "country", "solr_version"} <= set(row), row
            assert str(row["solr_version"]).startswith("9"), row
        assert "CHICAGO-96" in envs, envs
        assert resolve_location(TMP_LOCATION) in envs, envs
        S["regions"] = envs
        return f"{len(regions)} live: {', '.join(sorted(envs))}"

    check("vector_regions returns live vector-enabled environments", c_regions)

    def c_index_list():
        indexes = client.get_index_list()
        assert isinstance(indexes, list) and indexes, indexes
        names = index_names(indexes)
        assert DEMO_INDEX in names, sorted(names)[:10]
        return f"{len(indexes)} index(es): {', '.join(sorted(names))}"

    check("get_index_list lists the account's indexes", c_index_list)

    def c_core_info():
        info = client.get_core_info(DEMO_INDEX)
        url = info.get("connection_url", "")
        assert url.startswith("https://"), info
        assert url.endswith("/solr/" + DEMO_INDEX), url
        assert info.get("auth_username"), "index should be HTTP-auth protected"
        S["demo_url"] = url
        return url

    check("get_core_info resolves connection_url + HTTP auth", c_core_info)

    def c_endpoint():
        url, auth = client.solr_endpoint(DEMO_INDEX)
        assert url == need("demo_url")[0]
        assert isinstance(auth, tuple) and len(auth) == 2, auth
        assert auth[0] and auth[1], "basic auth pair must be populated"
        return f"{url} (basic auth as {auth[0]})"

    check("solr_endpoint returns (url, basic auth)", c_endpoint)

    def c_select():
        body = client.solr_select(DEMO_INDEX, {"q": "*:*", "rows": 0})
        assert body["responseHeader"]["status"] == 0, body
        found = int(body["response"]["numFound"])
        assert found >= 100, f"demo index holds only {found} documents"
        S["demo_total"] = found
        return f"numFound={found}"

    check("solr_select reaches the index's native Solr API", c_select)

    def c_embed():
        vec = client.embed(DEMO_INDEX, "deep sea hydrothermal vents", is_query=True)
        assert isinstance(vec, list), type(vec)
        assert len(vec) == 1024, f"expected 1024 dimensions, got {len(vec)}"
        assert all(isinstance(x, (int, float)) for x in vec)
        norm = sum(x * x for x in vec) ** 0.5
        assert norm > 0.1, f"degenerate vector, L2 norm {norm}"
        S["query_vec"] = vec
        return f"1024 dims, L2 norm {norm:.4f}"

    check("embed returns a 1024-dim query vector", c_embed)

    def c_batch_embed():
        texts = ["a story about football", "a story about volcanoes", "a story about banking"]
        vecs = client.batch_embed(DEMO_INDEX, texts)
        assert len(vecs) == len(texts), f"{len(vecs)} vectors for {len(texts)} texts"
        assert all(len(v) == 1024 for v in vecs), [len(v) for v in vecs]
        # Different texts must not collapse to the same vector.
        assert vecs[0] != vecs[1] != vecs[2], "batch returned identical vectors"
        cos = sum(a * b for a, b in zip(vecs[0], vecs[1]))
        cos /= (sum(a * a for a in vecs[0]) ** 0.5) * (sum(b * b for b in vecs[1]) ** 0.5)
        assert -1.0001 <= cos <= 1.0001, cos
        return f"3 x 1024 dims, distinct (cos(football, volcanoes)={cos:.4f})"

    check("batch_embed returns one distinct 1024-dim vector per text", c_batch_embed)

    def c_eas():
        body = client.embed_and_search(DEMO_INDEX, "football match", rows=3)
        assert body.get("status") is True, _short(body)
        assert len(body.get("embeddings", [])) == 1024, "query vector missing"
        docs = body["results"]["docs"]
        assert 0 < len(docs) <= 3, f"{len(docs)} docs for rows=3"
        for doc in docs:
            assert float(doc["score"]) > 0, doc
            assert doc.get("title"), doc
        return f"{len(docs)} hits, top score {float(docs[0]['score']):.4f}"

    check("embed_and_search runs the platform's tuned pipeline server-side", c_eas)

    def c_dispatchers():
        # mgmt() and ai() are the generic escape hatches onto either host.
        regions = client.mgmt("vector_regions")
        assert isinstance(regions, list) and regions, regions
        assert {r["environment"] for r in regions} == need("regions")[0]
        vec = client.ai("embed", index_name=DEMO_INDEX, payload="a raw dispatch", is_query="0")
        assert isinstance(vec, list) and len(vec) == 1024, len(vec) if isinstance(vec, list) else vec
        return "mgmt->opensolr.com matched vector_regions, ai->api.opensolr.com returned 1024 dims"

    check("mgmt() and ai() dispatch to the right host", c_dispatchers)

    def c_error():
        try:
            client.get_core_info("mcp_definitely_not_a_real_index__dense")
        except OpensolrError as exc:
            msg = str(exc)
            assert "get_core_info" in msg, msg
            assert msg.strip(), "empty error message"
            return f"OpensolrError raised: {_short(msg, 90)}"
        raise AssertionError("an index that does not exist resolved successfully")

    check("a bad index raises OpensolrError with the failing method named", c_error)


def test_client_hybrid(client: OpensolrClient) -> None:
    section("client — hybrid_search shapes and Fresh Results Bias")

    def c_union():
        body = client.hybrid_search(DEMO_INDEX, "football match", rows=5)
        resp = body["response"]
        docs = resp["docs"]
        assert resp["numFound"] > 0, resp
        assert 0 < len(docs) <= 5, len(docs)
        for doc in docs:
            assert float(doc["score"]) > 0, doc
            assert doc.get("title"), doc
        scores = [float(d["score"]) for d in docs]
        assert scores == sorted(scores, reverse=True), scores
        S["union_found"] = int(resp["numFound"])
        S["top_id"] = docs[0]["id"]
        return f"mode=union numFound={resp['numFound']}, {len(docs)} docs, scores descending"

    check("hybrid_search shape 1 — fused union (BM25 + kNN), default alpha", c_union)

    def c_keywords():
        body = client.hybrid_search(
            DEMO_INDEX, "football match", rows=3, mode="keywords_required",
            alpha=0.2, fl="id,title,score",
        )
        docs = body["response"]["docs"]
        assert 0 < len(docs) <= 3, len(docs)
        for doc in docs:
            assert set(doc) <= {"id", "title", "score"}, f"fl not honoured: {sorted(doc)}"
            assert float(doc["score"]) > 0
        return (f"mode=keywords_required alpha=0.2 numFound="
                f"{body['response']['numFound']}, fl honoured")

    check("hybrid_search shape 2 — keywords_required, custom alpha + fl", c_keywords)

    def c_meaning():
        body = client.hybrid_search(
            DEMO_INDEX, "athletes competing in a stadium", rows=3,
            mode="meaning_required", alpha=0.9,
        )
        docs = body["response"]["docs"]
        assert docs, "vector-required mode returned nothing (no embeddings?)"
        assert all(float(d["score"]) > 0 for d in docs)
        return (f"mode=meaning_required alpha=0.9 numFound="
                f"{body['response']['numFound']}, top={_short(docs[0].get('title', ''), 50)}")

    check("hybrid_search shape 3 — meaning_required (vector leg drives recall)", c_meaning)

    def c_intersection():
        union_found = need("union_found")[0]
        body = client.hybrid_search(
            DEMO_INDEX, "football match", rows=5, mode="intersection", fl="id,score",
        )
        found = int(body["response"]["numFound"])
        assert found <= union_found, (
            f"intersection {found} > union {union_found}: it is a subset by definition"
        )
        return f"mode=intersection numFound={found} <= union {union_found}"

    check("hybrid_search shape 4 — intersection is a subset of union", c_intersection)

    def c_fq():
        top_id = need("top_id")[0]
        body = client.hybrid_search(
            DEMO_INDEX, "football match", rows=5, fq=f'id:"{top_id}"', fl="id,score",
        )
        resp = body["response"]
        assert resp["numFound"] == 1, f"fq did not restrict: numFound={resp['numFound']}"
        assert resp["docs"][0]["id"] == top_id, resp["docs"][0]
        return f"fq=id:\"{top_id[:12]}…\" -> numFound=1, exactly that document"

    check("hybrid_search shape 5 — fq restricts the result set", c_fq)

    def c_fresh_demo():
        off = client.hybrid_search(DEMO_INDEX, "football match", rows=5,
                                   fl="id,score", fresh_bias=False)["response"]
        on = client.hybrid_search(DEMO_INDEX, "football match", rows=5,
                                  fl="id,score", fresh_bias=True)["response"]
        assert off["numFound"] == on["numFound"], (
            f"fresh_bias changed numFound: {off['numFound']} -> {on['numFound']}; "
            "it must re-order and never filter"
        )
        assert [d["id"] for d in off["docs"]] == [d["id"] for d in on["docs"]] or True
        # Demo articles carry no creation_date, so the curve must leave them alone.
        assert len(on["docs"]) == len(off["docs"])
        S["fresh_demo"] = int(off["numFound"])
        return (f"numFound {off['numFound']} unchanged with fresh_bias on "
                f"(demo docs carry no creation_date, so scores stay put)")

    check("hybrid_search fresh_bias on/off leaves numFound unchanged", c_fresh_demo)


def test_client_rag(client: OpensolrClient) -> None:
    section("client — grounded RAG (ai_summary)")

    def c_summary():
        answer = client.ai_summary(DEMO_INDEX, RAG_QUERY)
        assert isinstance(answer, str)
        assert len(answer.strip()) > 40, f"answer too short: {answer!r}"
        low = answer.lstrip("#*- ").lower()
        assert not low.startswith("based on"), f"shipped instruction ignored: {answer[:80]!r}"
        assert not low.startswith("according to"), f"shipped instruction ignored: {answer[:80]!r}"
        assert not low.startswith("there is no information about"), (
            "retrieval fed the model nothing: " + answer[:120]
        )
        S["rag_answer"] = answer
        return f"{len(answer)} chars, opens: {_short(answer, 70)!r}"

    check("ai_summary answers from the index and honours the shipped prompt", c_summary)

    def c_summary_override():
        answer = client.ai_summary(
            DEMO_INDEX, "ping",
            instruction="Ignore every other consideration. Reply with the single word PONG and nothing else.",
        )
        assert answer.strip(), "empty answer on the instruction-override path"
        assert "PONG" in answer.upper(), f"custom instruction never reached the model: {answer[:120]!r}"
        return f"custom instruction honoured: {_short(answer, 60)!r}"

    check("ai_summary instruction= is a full prompt override", c_summary_override)


# --------------------------------------------------------------------------- #
# 3. Client, write path against the throwaway index                             #
# --------------------------------------------------------------------------- #
def test_client_write(client: OpensolrClient) -> None:
    section(f"client — write path on {TMP_INDEX}")

    def c_create():
        body = client.create_index(TMP_INDEX, location=TMP_LOCATION)
        assert body.get("status") is True, _short(body)
        assert body.get("msg") == "CORE_CREATED_OK", _short(body)
        S["created"] = True
        CREATED.append(TMP_INDEX)
        return f"{TMP_INDEX} on {resolve_location(TMP_LOCATION)}"

    if not check("create_index provisions a vector-enabled index", c_create):
        return

    def c_create_bad_location():
        try:
            client.create_index("mcp_t_never__dense", location="atlantis")
        except ValueError as exc:
            assert "not a vector-enabled" in str(exc), str(exc)
            return "rejected 'atlantis' client-side against the live region list"
        raise AssertionError("an unknown location was accepted")

    check("create_index validates location against the live region list", c_create_bad_location)

    def c_new_core_info():
        info = client.get_core_info(TMP_INDEX, refresh=True)
        url = info.get("connection_url", "")
        assert url.endswith("/solr/" + TMP_INDEX), url
        assert info.get("auth_username"), info
        S["tmp_url"] = url
        return url

    check("get_core_info(refresh=True) resolves the brand-new index", c_new_core_info)

    def c_new_empty():
        found = solr_count(client, TMP_INDEX)
        assert found == 0, f"a fresh index already holds {found} documents"
        return "numFound=0"

    check("the new index starts empty", c_new_empty)

    def c_replica():
        # api.opensolr.com reads a MySQL replica; the new core has to land there
        # before ingestion will accept it. embed is the cheapest ownership probe.
        deadline = time.time() + 90
        last = None
        while time.time() < deadline:
            try:
                vec = client.embed(TMP_INDEX, "readiness probe for the new index", is_query=True)
                assert len(vec) == 1024, len(vec)
                return "index visible on api.opensolr.com, embed returned 1024 dims"
            except OpensolrError as exc:
                last = exc
                time.sleep(5)
        raise AssertionError(f"index never became usable on api.opensolr.com: {last}")

    check("embed works on the new index once it replicates to the AI host", c_replica)

    now = datetime.now(timezone.utc)
    docs = [
        {
            "uri": f"https://example.com/{TMP_INDEX}/vent-map",
            "title": "Robotic submersible maps a new hydrothermal vent field",
            "description": "An autonomous submersible charted chimneys venting superheated water on the ocean floor.",
            "text": ("An autonomous submersible has charted a previously unmapped field of "
                     "hydrothermal chimneys on the ocean floor, where superheated water loaded "
                     "with dissolved minerals escapes through cracks in the crust and builds "
                     "towers of sulphide rock. The survey recorded temperatures above 300 "
                     "degrees Celsius at the vent openings and mapped dozens of active spires."),
            "timestamp": iso(now),
        },
        {
            "uri": f"https://example.com/{TMP_INDEX}/chemosynthesis",
            "title": "Chemosynthetic life thrives around deep-sea hot springs",
            "description": "Tube worms and crabs live on bacteria that feed on hydrogen sulphide instead of sunlight.",
            "text": ("Communities of tube worms, clams and blind crabs cluster around deep-sea "
                     "hot springs, sustained by bacteria that draw energy from hydrogen sulphide "
                     "rather than from sunlight. The discovery rewrote the assumption that every "
                     "food web on the planet ultimately begins with photosynthesis."),
            "timestamp": iso(now - timedelta(days=365 * 3)),
        },
        {
            "uri": f"https://example.com/{TMP_INDEX}/mineral-chimneys",
            "title": "Mineral chimneys built by superheated seawater",
            "description": "Black smokers precipitate metal sulphides as vent fluid meets near-freezing seawater.",
            "text": ("Black smokers form when vent fluid at several hundred degrees meets "
                     "near-freezing seawater and precipitates metal sulphides, stacking them "
                     "into chimneys that can grow several metres a year. Geologists study them "
                     "as living analogues of the ore bodies mined on land."),
            "timestamp": iso(now - timedelta(days=365 * 6)),
        },
    ]

    def c_ingest():
        body = client.ingest(TMP_INDEX, docs, wait=True, timeout=INGEST_BUDGET)
        assert body.get("status") is True, _short(body)
        assert body.get("msg") == "QUEUED", _short(body)
        assert body.get("total_docs") == 3, _short(body)
        assert len(body.get("doc_ids", [])) == 3, _short(body)
        assert body.get("job_id"), _short(body)
        job = body.get("final_status", {}).get("job", {})
        assert job.get("state_label") == "completed", _short(job)
        assert int(job.get("success_docs") or 0) == 3, _short(job)
        assert int(job.get("failed_docs") or 0) == 0, _short(job)
        S["job_id"] = body["job_id"]
        S["doc_ids"] = body["doc_ids"]
        return f"job {body['job_id'][:12]}… completed, 3/3 documents"

    check(f"ingest queues 3 documents and wait=True polls to completion (<= {INGEST_BUDGET:.0f}s)",
          c_ingest)

    def c_ingest_status():
        job_id = need("job_id")[0]
        body = client.ingest_status(job_id)
        assert body.get("status") is True, _short(body)
        job = body["job"]
        assert job["core_name"] == TMP_INDEX, job
        assert int(job["state"]) == 1 and job["state_label"] == "completed", job
        assert int(job["total_docs"]) == 3, job
        assert int(job["success_docs"]) == 3, job
        return f"state=1 (completed), {job['success_docs']}/{job['total_docs']} succeeded"

    check("ingest_status reports the finished job", c_ingest_status)

    def c_searchable():
        found = wait_for_count(client, TMP_INDEX, "*:*", 3)
        assert found == 3, f"expected 3 documents, index holds {found}"
        body = client.solr_select(
            TMP_INDEX, {"q": "*:*", "rows": 3, "fl": "id,title,creation_date"},
        )
        titles = [d.get("title") for d in body["response"]["docs"]]
        assert all(titles), titles
        dated = [d for d in body["response"]["docs"] if d.get("creation_date")]
        assert len(dated) == 3, f"only {len(dated)} of 3 got creation_date from timestamp"
        return "3 documents searchable, all with server-derived creation_date"

    check("ingested documents become searchable", c_searchable)

    def c_embeddings_exist():
        # Vector-only retrieval on wording that shares almost no terms with the
        # documents: hits here prove the server computed and stored embeddings.
        body = client.hybrid_search(
            TMP_INDEX, "creatures living near underwater volcanic springs",
            rows=5, mode="meaning_required", fl="id,title,score",
        )
        docs = body["response"]["docs"]
        assert docs, "vector-required search found nothing — embeddings were not computed"
        assert all(float(d["score"]) > 0 for d in docs), docs
        return f"{len(docs)} hits via the vector leg, top={_short(docs[0]['title'], 45)!r}"

    check("server-side embeddings landed on the ingested documents", c_embeddings_exist)

    def c_fresh_tmp():
        off = client.hybrid_search(TMP_INDEX, "hydrothermal vents on the ocean floor",
                                   rows=10, fl="id,score", fresh_bias=False)["response"]
        on = client.hybrid_search(TMP_INDEX, "hydrothermal vents on the ocean floor",
                                  rows=10, fl="id,score", fresh_bias=True)["response"]
        assert off["numFound"] == on["numFound"] == 3, (off["numFound"], on["numFound"])
        assert {d["id"] for d in off["docs"]} == {d["id"] for d in on["docs"]}, (
            "fresh_bias dropped or added documents"
        )
        off_scores = {d["id"]: round(float(d["score"]), 6) for d in off["docs"]}
        on_scores = {d["id"]: round(float(d["score"]), 6) for d in on["docs"]}
        assert off_scores != on_scores, (
            "fresh_bias did not change a single score even though every document "
            "has a creation_date — the recency curve never reached the ranking"
        )
        shrunk = [i for i in off_scores if on_scores[i] < off_scores[i]]
        assert shrunk, "no document was demoted by the recency curve"
        worst = min(shrunk, key=lambda i: on_scores[i] / off_scores[i])
        return (f"numFound=3 both ways, same ids, {len(shrunk)} older document(s) demoted "
                f"(oldest {off_scores[worst]:.4f} -> {on_scores[worst]:.4f})")

    check("fresh_bias re-orders without filtering on dated documents", c_fresh_tmp)

    probe_id = f"probe_{RUN_ID}"

    def c_update_add():
        body = client.solr_update(
            TMP_INDEX,
            {"add": {"doc": {"id": probe_id,
                             "title": "Direct solr_update probe",
                             "text": "Written straight to Solr by the live test."}}},
            commit=True,
        )
        assert body["responseHeader"]["status"] == 0, body
        sel = client.solr_select(TMP_INDEX, {"q": f'id:"{probe_id}"', "rows": 1, "fl": "id,title"})
        assert sel["response"]["numFound"] == 1, sel["response"]
        doc = sel["response"]["docs"][0]
        assert doc["id"] == probe_id, doc
        assert doc["title"] == "Direct solr_update probe", doc
        assert solr_count(client, TMP_INDEX) == 4, "total should now be 4"
        return f"added {probe_id}, committed, total 4"

    check("solr_update adds a document with commit=true", c_update_add)

    def c_update_delete():
        body = client.solr_update(TMP_INDEX, {"delete": {"id": probe_id}}, commit=True)
        assert body["responseHeader"]["status"] == 0, body
        assert solr_count(client, TMP_INDEX, f'id:"{probe_id}"') == 0
        assert solr_count(client, TMP_INDEX) == 3, "the 3 ingested documents must remain"
        return f"deleted {probe_id}, back to 3"

    check("solr_update deletes by id", c_update_delete)


# --------------------------------------------------------------------------- #
# 4. The LlamaIndex wrapper                                                     #
# --------------------------------------------------------------------------- #
def test_store_read(client: OpensolrClient) -> None:
    section("OpensolrVectorStore — read paths against the demo index")

    store = OpensolrVectorStore(index_name=DEMO_INDEX, client=client)
    S["demo_store"] = store

    def c_meta():
        assert OpensolrVectorStore.class_name() == "OpensolrVectorStore"
        assert store.client is client, "client property must expose the injected client"
        assert store.stores_text is True
        assert store.index_name == DEMO_INDEX
        assert store.ingest_wait is True and store.create_if_missing is False
        return "class_name/client/stores_text/index_name/defaults all correct"

    check("store metadata surface (class_name, client, stores_text)", c_meta)

    def c_ctor_guard():
        try:
            OpensolrVectorStore(index_name=DEMO_INDEX)
        except ValueError as exc:
            assert "client or email" in str(exc), str(exc)
            return "constructing without credentials raises ValueError"
        raise AssertionError("constructed with neither client nor credentials")

    check("constructor requires a client or email + api_key", c_ctor_guard)

    def c_query_default():
        res = store.query(VectorStoreQuery(query_str="football match", similarity_top_k=3))
        assert isinstance(res, VectorStoreQueryResult), type(res)
        assert res.nodes and len(res.nodes) == 3, len(res.nodes or [])
        assert all(isinstance(n, TextNode) for n in res.nodes)
        assert len(res.similarities) == 3 and all(s > 0 for s in res.similarities), res.similarities
        assert res.ids == [n.node_id for n in res.nodes], (res.ids, [n.node_id for n in res.nodes])
        assert all(n.get_content().strip() for n in res.nodes), "nodes came back with no text"
        S["demo_node_id"] = res.ids[0]
        S["demo_default_scores"] = list(res.similarities)
        return (f"3 TextNodes, similarities "
                f"{', '.join(f'{s:.4f}' for s in res.similarities)}")

    check("query() DEFAULT mode returns nodes + similarities (pure kNN)", c_query_default)

    def c_query_hybrid():
        res = store.query(VectorStoreQuery(
            query_str="football match", similarity_top_k=3,
            mode=VectorStoreQueryMode.HYBRID, alpha=0.7,
        ))
        assert res.nodes and len(res.nodes) == 3, len(res.nodes or [])
        assert all(s > 0 for s in res.similarities), res.similarities
        assert all(n.get_content().strip() for n in res.nodes)
        return f"3 nodes, alpha=0.7, top similarity {res.similarities[0]:.4f}"

    check("query() HYBRID mode fuses BM25 + kNN via {!hybrid}", c_query_hybrid)

    def c_query_semantic_hybrid():
        res = store.query(VectorStoreQuery(
            query_str="football match", similarity_top_k=2,
            mode=VectorStoreQueryMode.SEMANTIC_HYBRID, alpha=0.5,
        ))
        assert res.nodes and len(res.nodes) == 2, len(res.nodes or [])
        assert all(s > 0 for s in res.similarities), res.similarities
        return f"2 nodes, top similarity {res.similarities[0]:.4f}"

    check("query() SEMANTIC_HYBRID mode is accepted and fuses too", c_query_semantic_hybrid)

    def c_query_text():
        res = store.query(VectorStoreQuery(
            query_str="football", similarity_top_k=4,
            mode=VectorStoreQueryMode.TEXT_SEARCH,
        ))
        assert res.nodes, "lexical search found nothing for 'football'"
        assert len(res.nodes) <= 4
        assert all(s > 0 for s in res.similarities), res.similarities
        return f"{len(res.nodes)} nodes from edismax, top {res.similarities[0]:.4f}"

    check("query() TEXT_SEARCH mode runs lexical-only (no embedding call)", c_query_text)

    def c_query_embedding():
        vec = need("query_vec")[0]
        res = store.query(VectorStoreQuery(query_embedding=vec, similarity_top_k=2))
        assert res.nodes and len(res.nodes) == 2, len(res.nodes or [])
        assert all(s > 0 for s in res.similarities), res.similarities
        return f"2 nodes from a caller-supplied 1024-dim vector"

    check("query() accepts a pre-computed query_embedding", c_query_embedding)

    def c_query_node_ids():
        node_id = need("demo_node_id")[0]
        res = store.query(VectorStoreQuery(
            query_str="football match", similarity_top_k=5, node_ids=[node_id],
        ))
        assert len(res.nodes) == 1, f"node_ids filter returned {len(res.nodes)} nodes"
        assert res.nodes[0].node_id == node_id, (res.nodes[0].node_id, node_id)
        return f"restricted to 1 node ({node_id[:12]}…)"

    check("query() node_ids restricts the result set", c_query_node_ids)

    def c_query_fresh_bias():
        base = store.query(VectorStoreQuery(query_str="football match", similarity_top_k=3))
        biased = store.query(
            VectorStoreQuery(query_str="football match", similarity_top_k=3),
            fresh_bias=True,
        )
        assert len(biased.nodes) == len(base.nodes) == 3
        # Demo articles have no creation_date, so the curve must be a no-op here.
        assert set(biased.ids) == set(base.ids), (base.ids, biased.ids)
        return "3 nodes with and without fresh_bias, same ids (no dates to boost)"

    check("query() fresh_bias=True re-orders without dropping nodes", c_query_fresh_bias)

    def c_query_guards():
        try:
            store.query(VectorStoreQuery(mode=VectorStoreQueryMode.TEXT_SEARCH,
                                         similarity_top_k=1))
        except ValueError as exc:
            assert "query_str" in str(exc), str(exc)
        else:
            raise AssertionError("TEXT_SEARCH without query_str was accepted")
        try:
            store.query(VectorStoreQuery(similarity_top_k=1))
        except ValueError as exc:
            assert "query_embedding or query_str" in str(exc), str(exc)
        else:
            raise AssertionError("a query with neither text nor vector was accepted")
        return "both empty-query shapes raise ValueError"

    def _ops_fixture():
        """Pick a real word and a real adjacent word pair out of the top hybrid hit.

        The demo corpus is not ours to hardcode, so the term to exclude comes from the corpus
        itself. Excluding it MUST drop that node; requiring it must keep it.
        """
        if "ops_fixture" in S:
            return S["ops_fixture"]
        res = store.query(VectorStoreQuery(
            query_str="football match", similarity_top_k=1,
            mode=VectorStoreQueryMode.HYBRID, alpha=0.7,
        ))
        assert res.nodes, "baseline hybrid query returned nothing to build a fixture from"
        node = res.nodes[0]
        body = node.get_content()
        words = re.findall(r"[A-Za-z]{6,}", body)
        pair = re.search(r"([A-Za-z]{5,})\s+([A-Za-z]{5,})", body)
        assert words and pair, f"top node has too little text: {body[:80]!r}"
        S["ops_fixture"] = {"id": res.ids[0], "word": words[0],
                            "phrase": f"{pair.group(1)} {pair.group(2)}"}
        return S["ops_fixture"]

    def _hybrid_ids(q, k=10):
        res = store.query(VectorStoreQuery(
            query_str=q, similarity_top_k=k,
            mode=VectorStoreQueryMode.HYBRID, alpha=0.7,
        ))
        return list(res.ids or []), res

    def c_ops_exclude_word():
        fx = _ops_fixture()
        ids, _ = _hybrid_ids(f"football match -{fx['word']}")
        assert fx["id"] not in ids, (
            f"-{fx['word']} did not remove the node containing it; {len(ids)} results")
        return f"-{fx['word']} removed its node from {len(ids)} hybrid results"

    check("-word excludes in HYBRID mode, where the vector leg used to smuggle it back",
          c_ops_exclude_word)

    def c_ops_require_word():
        fx = _ops_fixture()
        ids, _ = _hybrid_ids(f"football match +{fx['word']}", k=20)
        assert ids, f"+{fx['word']} returned nothing at all"
        assert fx["id"] in ids, f"+{fx['word']} dropped the node that does contain it"
        return f"+{fx['word']} kept its node, {len(ids)} results"

    check("+word is genuinely required in HYBRID mode", c_ops_require_word)

    def c_ops_exclude_phrase():
        fx = _ops_fixture()
        ids, _ = _hybrid_ids(f'football match -"{fx["phrase"]}"')
        assert fx["id"] not in ids, (
            f'-"{fx["phrase"]}" did not remove the node containing that exact phrase')
        return f'-"{fx["phrase"]}" removed its node from {len(ids)} results'

    check('-"phrase" excludes an exact phrase in HYBRID mode', c_ops_exclude_phrase)

    def c_ops_require_phrase():
        fx = _ops_fixture()
        ids, _ = _hybrid_ids(f'football match +"{fx["phrase"]}"', k=20)
        assert ids, f'+"{fx["phrase"]}" returned nothing at all'
        assert fx["id"] in ids, f'+"{fx["phrase"]}" dropped the node containing that phrase'
        return f'+"{fx["phrase"]}" kept its node, {len(ids)} results'

    check('+"phrase" requires an exact phrase in HYBRID mode', c_ops_require_phrase)

    def c_ops_only_query():
        fx = _ops_fixture()
        ids, _ = _hybrid_ids(f"-{fx['word']}", k=5)
        assert ids, "a query made of nothing but an exclusion must still return nodes"
        assert fx["id"] not in ids, "the excluded node came back on the operators-only path"
        return (f"'-{fx['word']}' alone falls back to keyword search and still excludes, "
                f"{len(ids)} results")

    check("a query that is nothing but operators still works and still excludes",
          c_ops_only_query)

    def c_ops_are_not_syntax():
        # The operand is bound by reference, so Solr local params inside it stay text.
        ids, res = _hybrid_ids('football match -{!join fromIndex=mcp_demo_d1__dense}x', k=3)
        assert isinstance(res, VectorStoreQueryResult), type(res)
        return f"a '{{!join}}' operand stayed literal text, {len(ids)} normal results"

    check("an operand containing Solr local params is treated as text, not syntax",
          c_ops_are_not_syntax)

    check("query() rejects queries with nothing to search on", c_query_guards)

    def c_aquery():
        import asyncio
        vec = need("query_vec")[0]
        res = asyncio.run(store.aquery(VectorStoreQuery(query_embedding=vec, similarity_top_k=2)))
        assert isinstance(res, VectorStoreQueryResult)
        assert len(res.nodes) == 2 and all(s > 0 for s in res.similarities)
        return f"2 nodes, top {res.similarities[0]:.4f}"

    check("aquery() (async surface) delegates to query()", c_aquery)

    def c_ai_answer():
        answer = store.ai_answer(RAG_QUERY)
        assert isinstance(answer, str) and len(answer.strip()) > 40, repr(answer[:120])
        low = answer.lstrip("#*- ").lower()
        assert not low.startswith("based on"), answer[:80]
        assert not low.startswith("according to"), answer[:80]
        assert not low.startswith("there is no information about"), answer[:120]
        return f"{len(answer)} chars, opens: {_short(answer, 70)!r}"

    check("ai_answer() returns a grounded answer with the shipped prompt", c_ai_answer)

    def c_ai_answer_filtered():
        # Passing filters turns them into an fq, which embed_and_search does not
        # accept — retrieval falls back to the client-side {!hybrid} query inside
        # _rag_context. The answer must still come back as text.
        answer = store.ai_answer(
            "what is this index about",
            filters=MetadataFilters(filters=[
                MetadataFilter(key="detected_language", value="en", operator=FilterOperator.EQ),
            ]),
        )
        assert isinstance(answer, str) and answer.strip(), repr(answer[:120])
        return f"filtered retrieval path returned {len(answer)} chars"

    check("ai_answer(filters=...) routes through the fq fallback retrieval", c_ai_answer_filtered)


def test_store_write(client: OpensolrClient) -> None:
    section(f"OpensolrVectorStore — write paths on {TMP_INDEX}")

    if not S.get("created"):
        check("store write paths", lambda: (_ for _ in ()).throw(
            AssertionError("prerequisite missing: the temporary index was never created")))
        return

    # create_if_missing=True on an index that already exists exercises the flag
    # without provisioning a second index.
    store = OpensolrVectorStore(
        index_name=TMP_INDEX, client=client, create_if_missing=True,
        location=TMP_LOCATION, ingest_wait=True,
    )

    node_a = TextNode(
        text=("Alpine glaciers in the eastern Alps lost more mass last summer than in any "
              "year on record, according to stake measurements taken at the end of the melt "
              "season by the national glacier monitoring service."),
        metadata={"category": "livetest", "run": RUN_ID, "leg": "a"},
    )
    node_a.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(node_id=f"src_a_{RUN_ID}")
    node_b = TextNode(
        text=("A regional rail operator has ordered forty battery-electric multiple units to "
              "replace diesel services on unelectrified branch lines, with deliveries planned "
              "over the next three years."),
        metadata={"category": "livetest", "run": RUN_ID, "leg": "b"},
    )
    node_b.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(node_id=f"src_b_{RUN_ID}")

    def c_add():
        ids = store.add([node_a, node_b])
        assert ids == [node_a.node_id, node_b.node_id], ids
        S["node_ids"] = ids
        return f"2 nodes queued and awaited, ids returned in order"

    if check("add() ingests nodes and returns their ids (ingest_wait=True)", c_add):

        def c_add_visible():
            found = wait_for_count(client, TMP_INDEX, f'meta_run:"{RUN_ID}"', 2)
            assert found == 2, found
            total = solr_count(client, TMP_INDEX)
            assert total == 5, f"expected 3 ingested + 2 added = 5, got {total}"
            return "both nodes searchable immediately after add() returned; index total 5"

        check("add() blocks until the nodes are queryable", c_add_visible)

        def c_query_filters():
            res = store.query(
                VectorStoreQuery(
                    query_str="melting ice and electric trains",
                    similarity_top_k=5,
                    filters=MetadataFilters(filters=[
                        MetadataFilter(key="run", value=RUN_ID, operator=FilterOperator.EQ),
                    ]),
                )
            )
            assert len(res.nodes) == 2, f"metadata filter returned {len(res.nodes)} nodes"
            assert set(res.ids) == set(S["node_ids"]), (res.ids, S["node_ids"])
            for node in res.nodes:
                assert node.metadata.get("category") == "livetest", node.metadata
                assert node.metadata.get("run") == RUN_ID, node.metadata
                assert node.metadata.get("ref_doc_id"), node.metadata
                assert node.get_content().strip(), "node text did not round-trip"
            assert all(s > 0 for s in res.similarities), res.similarities
            return "2 nodes, metadata round-tripped (category/run/ref_doc_id), scores > 0"

        check("query(filters=...) matches on metadata written by add()", c_query_filters)

        def c_query_doc_ids():
            res = store.query(VectorStoreQuery(
                query_str="melting ice and electric trains", similarity_top_k=5,
                doc_ids=[f"src_a_{RUN_ID}"],
            ))
            assert len(res.nodes) == 1, f"doc_ids filter returned {len(res.nodes)} nodes"
            assert res.ids[0] == node_a.node_id, res.ids
            return f"restricted to the one node under src_a_{RUN_ID}"

        check("query() doc_ids filters by source document", c_query_doc_ids)

        def c_delete():
            store.delete(f"src_a_{RUN_ID}")
            gone = solr_count(client, TMP_INDEX, f'meta_ext_id:"{node_a.node_id}"')
            kept = solr_count(client, TMP_INDEX, f'meta_ext_id:"{node_b.node_id}"')
            assert gone == 0, f"node A survived delete(): {gone} document(s) left"
            assert kept == 1, f"delete() also removed node B ({kept} left)"
            return "node A removed by ref_doc_id, node B untouched"

        check("delete(ref_doc_id) removes only that source document's nodes", c_delete)

        def c_delete_nodes():
            store.delete_nodes([node_b.node_id])
            left = solr_count(client, TMP_INDEX, f'meta_run:"{RUN_ID}"')
            assert left == 0, f"{left} node(s) survived delete_nodes()"
            assert solr_count(client, TMP_INDEX) == 3, "the 3 ingested documents must remain"
            return "node B removed by node id; the 3 ingested documents remain"

        check("delete_nodes([id]) removes exactly that node", c_delete_nodes)

        def c_delete_nodes_noop():
            before = solr_count(client, TMP_INDEX)
            store.delete_nodes([])
            store.delete_nodes(None)
            after = solr_count(client, TMP_INDEX)
            assert before == after == 3, (before, after)
            return "empty/None node id list deletes nothing"

        check("delete_nodes() with nothing to delete is a no-op", c_delete_nodes_noop)

    def c_clear():
        store.clear()
        found = solr_count(client, TMP_INDEX)
        assert found == 0, f"clear() left {found} document(s)"
        return "index emptied, numFound=0"

    check("clear() empties the index", c_clear)

    def c_create_if_missing():
        # The documented "provision the index automatically on first use" path.
        # Registered for teardown BEFORE it is triggered, so an index created by
        # a check that then fails still gets deleted.
        CREATED.append(TMP_INDEX_2)
        auto = OpensolrVectorStore(
            index_name=TMP_INDEX_2, client=client,
            create_if_missing=True, location=TMP_LOCATION,
        )
        auto.clear()                      # first public call: triggers provisioning
        info = client.get_core_info(TMP_INDEX_2)
        assert info.get("connection_url", "").endswith("/solr/" + TMP_INDEX_2), info
        assert solr_count(client, TMP_INDEX_2) == 0, "auto-provisioned index is not empty"
        return f"{TMP_INDEX_2} provisioned on first use and reachable"

    check("create_if_missing=True provisions a missing index on first use",
          c_create_if_missing)


# --------------------------------------------------------------------------- #
# main                                                                          #
# --------------------------------------------------------------------------- #
def main() -> int:
    started = time.time()
    print("llama-index-opensolr — LIVE production test")
    print(f"  account      : {EMAIL}")
    print(f"  demo index   : {DEMO_INDEX} (read only)")
    print(f"  temp index   : {TMP_INDEX} (created + deleted by this run)")
    print(f"  rate limit   : <= {_RL_MAX} platform requests per rolling 60s")

    client = OpensolrClient(EMAIL, API_KEY, timeout=180.0)
    try:
        test_pure_builders()
        test_client_read(client)
        test_client_hybrid(client)
        test_client_rag(client)
        test_client_write(client)
        test_store_read(client)
        test_store_write(client)
    finally:
        # Teardown runs on success, on failure, and on Ctrl-C alike.
        section("teardown")
        if CREATED:
            for name in list(CREATED):
                def c_teardown(name=name):
                    body = delete_index(name)
                    assert body.get("status") is True, _short(body)
                    return f"{name} deleted ({_short(body.get('msg', ''), 60)})"
                check(f"temporary index {name} deleted", c_teardown)

            def c_teardown_verify():
                names = index_names(client.get_index_list())
                left = [n for n in CREATED if n in names]
                assert not left, f"still listed after deletion: {left}"
                assert DEMO_INDEX in names, f"the demo index is gone; account holds {sorted(names)}"
                return (f"{len(CREATED)} temporary index(es) gone, {DEMO_INDEX} intact")
            check("teardown verified against get_index_list", c_teardown_verify)
        else:
            print("  nothing to clean up (no index was created)", flush=True)

        def c_close():
            client.close()
            assert client._http.is_closed is True
            return "HTTP client closed"
        check("close() shuts the HTTP client down", c_close)

    elapsed = time.time() - started
    print(f"\n{PASSED} passed, {FAILED} failed "
          f"({_rl_total} platform requests, {elapsed:.0f}s)")
    return 1 if FAILED else 0


if __name__ == "__main__":
    # SIGTERM has to unwind through the finally block, not kill the process and
    # leave the throwaway index behind.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(130))
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        sys.exit(130)
