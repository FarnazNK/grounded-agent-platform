"""Retrieval pipeline: dense + sparse fusion.

Pure dense retrieval is fragile for keyword-heavy queries (acronyms, IDs,
exact product names). We add BM25 over the same corpus and fuse with
reciprocal rank fusion (RRF) — cheap, robust, no extra model needed.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import dataclass

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from rag_agent.config import get_settings
from rag_agent.observability import get_logger
from rag_agent.schemas import RetrievedChunk
from rag_agent.vectorstore import VectorStore

log = get_logger(__name__)


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + lowercase tokenization. Good enough for BM25."""
    return [t for t in text.lower().split() if t]


@dataclass
class HybridRetriever:
    """Combines dense vector search with BM25 over an in-memory corpus.

    The BM25 corpus is held in memory because (a) it's tiny relative to the
    vectors and (b) BM25Okapi has no persistent format. For larger corpora
    you'd swap this for OpenSearch or pgroonga — same interface.
    """

    store: VectorStore
    corpus: list[Document]

    def __post_init__(self) -> None:
        self._tokenized = [_tokenize(d.page_content) for d in self.corpus]
        self._bm25 = BM25Okapi(self._tokenized) if self._tokenized else None

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        settings = get_settings()
        dense = self.store.similarity_search(query, k=settings.top_k_dense)
        sparse = self._bm25_search(query, k=settings.top_k_sparse)

        fused = _reciprocal_rank_fusion([dense, sparse])
        filtered = [c for c in fused if c.score >= settings.min_relevance_score]
        top = filtered[: settings.top_k_final]

        log.info(
            "retrieval.complete",
            dense=len(dense),
            sparse=len(sparse),
            fused=len(fused),
            returned=len(top),
        )
        return top

    async def aretrieve(self, query: str) -> list[RetrievedChunk]:
        """Async retrieve for the serving path.

        Both halves of this are blocking in a way that matters under
        concurrency:

        - `store.similarity_search` makes a synchronous embeddings HTTP call
          and a Chroma query.
        - `BM25Okapi.get_scores` is pure-Python CPU work, O(corpus) per query.

        Neither has a native async API, so we hand the whole fused retrieve to
        a worker thread. The GIL still serializes the BM25 arithmetic, but the
        event loop stays free to accept connections and pump other sessions —
        which is the property we actually need. If BM25 ever becomes the
        bottleneck, the fix is a process pool or a real search backend
        (OpenSearch), not a different await.
        """
        return await asyncio.to_thread(self.retrieve, query)

    def _bm25_search(self, query: str, k: int) -> list[RetrievedChunk]:
        if not self._bm25:
            return []
        scores = self._bm25.get_scores(_tokenize(query))
        if not len(scores):
            return []

        # Min-max normalize so BM25 scores live in roughly the same range as
        # the dense relevance scores. Pure rank fusion below doesn't strictly
        # need this but it makes the per-source scores comparable for logs.
        s_min, s_max = float(scores.min()), float(scores.max())
        spread = s_max - s_min or 1.0

        ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        chunks: list[RetrievedChunk] = []
        for i in ranked_idx:
            doc = self.corpus[i]
            norm = (float(scores[i]) - s_min) / spread
            chunks.append(
                RetrievedChunk(
                    chunk_id=doc.metadata.get("chunk_id", str(i)),
                    content=doc.page_content,
                    source=doc.metadata.get("source", "unknown"),
                    score=norm,
                    metadata=doc.metadata,
                )
            )
        return chunks


def _reciprocal_rank_fusion(
    result_lists: list[list[RetrievedChunk]],
    k: int = 60,
) -> list[RetrievedChunk]:
    """Standard RRF. The constant `k=60` is the value from the original
    Cormack et al. paper — robust default, no need to tune."""
    fused_scores: dict[str, float] = defaultdict(float)
    seen: dict[str, RetrievedChunk] = {}

    for results in result_lists:
        for rank, chunk in enumerate(results):
            fused_scores[chunk.chunk_id] += 1.0 / (k + rank + 1)
            # Keep the first-seen instance but we'll overwrite the score below.
            seen.setdefault(chunk.chunk_id, chunk)

    # Normalize fused scores to [0, 1] for downstream filtering.
    if not fused_scores:
        return []
    max_score = max(fused_scores.values())

    out: list[RetrievedChunk] = []
    for cid, score in sorted(fused_scores.items(), key=lambda kv: kv[1], reverse=True):
        chunk = seen[cid].model_copy(update={"score": score / max_score})
        out.append(chunk)
    return out
