"""
rag_client.py — RAG knowledge base query client.

Queries a vector database of chunked & embedded FAQ documents.
Returns top-k chunks for the LLM to generate grounded answers.

Supported backends (set VECTOR_DB_BACKEND env var):
  mock     — built-in stub responses (development/testing)
  chroma   — ChromaDB (local or remote)
  pgvector — PostgreSQL + pgvector

Setup for Chroma (free, self-hosted):
  pip install chromadb sentence-transformers
  Set VECTOR_DB_URL=http://localhost:8000  (or leave blank for in-process)
  Set VECTOR_DB_COLLECTION=faq_docs
  Ingest documents: python scripts/ingest_faq.py --dir docs/faq/

Setup for pgvector:
  pip install psycopg2-binary pgvector
  Set VECTOR_DB_URL=postgresql://user:pass@host/db
  Set VECTOR_DB_COLLECTION=faq_chunks
"""

import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

RAG_TOP_K          = int(os.getenv("RAG_TOP_K", "5"))
VECTOR_DB_BACKEND  = os.getenv("VECTOR_DB_BACKEND", "mock")
COLLECTION         = os.getenv("VECTOR_DB_COLLECTION", "faq_docs")
EMBEDDING_MODEL    = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class RAGChunk:
    chunk_id:     str
    text:         str
    score:        float           # cosine similarity 0–1
    source_doc:   str
    section:      str
    product_type: Optional[str] = None
    heritage_brand: Optional[str] = None


@dataclass
class RAGResult:
    query:        str
    query_hash:   str = ""
    chunks:       list[RAGChunk] = field(default_factory=list)
    cache_hit:    bool = False
    answerable:   bool = True     # False if no chunk meets threshold
    confidence:   float = 0.0    # top-chunk score (0.0–1.0)
    context:      str = ""       # pre-formatted context string for fallback messages

    def context_for_llm(self) -> str:
        """Format retrieved chunks as a context block for the LLM prompt."""
        if not self.chunks:
            return "[No FAQ context available]"
        parts = []
        for i, c in enumerate(self.chunks, 1):
            parts.append(f"[Source {i}: {c.source_doc} — {c.section}]\n{c.text}")
        return "\n\n".join(parts)


# ── Mock FAQ data (development / testing) ─────────────────────────────────────

_MOCK_FAQ: list[dict] = [
    {
        "chunk_id": "faq-001",
        "text": ("The minimum retirement age for pension policies is currently 55, "
                 "rising to 57 in April 2028 under HMRC rules. You can take your pension "
                 "benefits earlier if you are in ill health, subject to scheme rules."),
        "source_doc": "pension_retirement_guide.pdf",
        "section": "Retirement Age",
        "product_type": "pension",
    },
    {
        "chunk_id": "faq-002",
        "text": ("Pension transfers to another registered pension scheme are permitted under "
                 "HMRC rules. A transfer value quotation is valid for 3 months. You will need "
                 "to complete a transfer request form. If the transfer value exceeds £30,000 "
                 "and you have any safeguarded benefits (such as a guaranteed annuity rate), "
                 "you must obtain independent financial advice before proceeding."),
        "source_doc": "pension_transfer_guide.pdf",
        "section": "Transfer Process",
        "product_type": "pension",
    },
    {
        "chunk_id": "faq-003",
        "text": ("The annual allowance for pension contributions in the current tax year is "
                 "£60,000 gross (or 100% of earnings if lower). If you exceed this, a tax "
                 "charge applies. The money purchase annual allowance (MPAA) of £10,000 applies "
                 "if you have flexibly accessed your pension savings."),
        "source_doc": "pension_annual_allowance.pdf",
        "section": "Annual Allowance",
        "product_type": "pension",
    },
    {
        "chunk_id": "faq-004",
        "text": ("To make a life assurance claim, please complete a claim notification form "
                 "available on our website or by contacting the claims team. You will need to "
                 "provide the original policy document, a certified copy of the death certificate, "
                 "and proof of identity for the claimant. Claims are typically assessed within 5 working days."),
        "source_doc": "life_claims_process.pdf",
        "section": "Making a Claim",
        "product_type": "life",
    },
    {
        "chunk_id": "faq-005",
        "text": ("You have a 30-day cooling-off period from the date your policy was issued during "
                 "which you may cancel your policy and receive a full refund of any premiums paid, "
                 "subject to a deduction for any risk that has been on cover."),
        "source_doc": "cancellation_rights.pdf",
        "section": "Cooling-off Period",
        "product_type": None,  # applies to all product types
    },
]


STOP_WORDS   = {"the", "is", "of", "and", "a", "to", "in", "it", "on", "for", "my", "your", "with", "what"}
INTENT_WORDS = {"process", "how", "timeline", "steps", "way", "method", "calculate", "charges", "fees", "value", "tax", "taxable"}

def _mock_score(query: str, chunk_text: str) -> float:
    """Weighted keyword-overlap score for mock mode (superior quality)."""
    q_words = set(re.findall(r'\w+', query.lower()))
    c_words = set(re.findall(r'\w+', chunk_text.lower()))
    
    # Stemming light
    def stem(w): return w[:-1] if w.endswith('s') and len(w)>4 else w
    q_stems = {stem(w) for w in q_words}
    c_stems = {stem(w) for w in c_words}
    
    overlap = q_stems & c_stems
    
    # Calculate weighted overlap
    score = 0.0
    for s in overlap:
        if s in INTENT_WORDS:
            score += 0.25  # High priority for intent markers
        elif s in STOP_WORDS:
            score += 0.02  # Minimal weight for noise
        else:
            score += 0.12  # Standard weight for nouns/entities
            
    # Base score of 0.3 + accumulated weight, capped at 0.95
    return min(0.95, 0.3 + score)


# ── Client ────────────────────────────────────────────────────────────────────

class RAGClient:
    _model_cache = {}

    def __init__(self, session_cache) -> None:
        self._cache  = session_cache
        self._backend = VECTOR_DB_BACKEND
        self._mock   = self._backend == "mock" or os.getenv("USE_MOCK_RAG", "false").lower() == "true"
        self._embedger = None
        if not self._mock:
            self._init_backend()

    def _get_embedger(self):
        """Singleton pattern for the embedding model to prevent reloading weights."""
        if EMBEDDING_MODEL not in RAGClient._model_cache:
            logger.info("Loading embedding model %s into memory...", EMBEDDING_MODEL)
            from sentence_transformers import SentenceTransformer
            RAGClient._model_cache[EMBEDDING_MODEL] = SentenceTransformer(EMBEDDING_MODEL)
        return RAGClient._model_cache[EMBEDDING_MODEL]

    def list_all_faqs(self) -> list[dict]:
        """Return all FAQs (for Knowledge Hub UI) by loading from the source document."""
        source_path = "docs/faq/pension_faqs.txt"
        faqs = []
        if os.path.exists(source_path):
            try:
                import json
                with open(source_path, "r") as f:
                    data = json.load(f)
                    raw_faqs = data.get("faqs", [])
                    # Normalize fields to match the RAGChunk expectations
                    for r in raw_faqs:
                        faqs.append({
                            "chunk_id": r.get("id") or r.get("chunk_id"),
                            "text": f"Question: {r.get('question','')}\nAnswer: {r.get('answer','')}",
                            "section": r.get("category") or r.get("section") or "General",
                            "source_doc": r.get("source_doc") or "pension_faqs.txt",
                            "product_type": r.get("product_type") or "all",
                            "heritage_brand": r.get("heritage_brand") or "all"
                        })
                    return faqs
            except Exception as e:
                logger.error("Failed to load FAQs from %s: %s", source_path, e)
        
        return _MOCK_FAQ

    def _init_backend(self) -> None:
        if self._backend == "chroma":
            try:
                import chromadb
                from chromadb.utils import embedding_functions
                
                db_url = os.getenv("VECTOR_DB_URL", "")
                if db_url.startswith("http") or db_url.startswith("https"):
                    import urllib.parse
                    parsed = urllib.parse.urlparse(db_url)
                    client = chromadb.HttpClient(host=parsed.hostname or "localhost", port=parsed.port or 8000)
                    logger.info("Chroma RAG client (HTTP) initialised")
                else:
                    path = os.getenv("CHROMA_PATH", "./chroma_data")
                    client = chromadb.PersistentClient(path=path)
                    logger.info("Chroma RAG client (Persistent) initialised at %s", path)

                ef = embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name=EMBEDDING_MODEL)
                self._collection = client.get_or_create_collection(COLLECTION, embedding_function=ef)
            except ImportError:
                logger.warning("chromadb not installed — falling back to mock RAG")
                self._mock = True
        elif self._backend == "pinecone":
            try:
                from pinecone import Pinecone
                
                api_key = os.getenv("PINECONE_API_KEY") or (st.secrets.get("PINECONE_API_KEY") if "st" in globals() else None)
                index_name = os.getenv("PINECONE_INDEX_NAME") or (st.secrets.get("PINECONE_INDEX_NAME") if "st" in globals() else None)
                
                if not api_key or not index_name:
                    logger.warning("Pinecone credentials missing — falling back to mock RAG")
                    self._mock = True
                    return

                self._pc = Pinecone(api_key=api_key)
                self._index = self._pc.Index(index_name)
                # Model is now lazy-loaded via _get_embedger()
                logger.info("Pinecone RAG client initialised (Index: %s)", index_name)
            except ImportError:
                logger.warning("pinecone-client or sentence-transformers not installed — falling back to mock RAG")
                self._mock = True
        elif self._backend == "pgvector":
            try:
                import psycopg2
                self._pg_conn = psycopg2.connect(os.environ["VECTOR_DB_URL"])
                logger.info("pgvector RAG client initialised")
            except ImportError:
                logger.warning("psycopg2 not installed — falling back to mock RAG")
                self._mock = True
        else:
            logger.warning("Unknown VECTOR_DB_BACKEND %r — using mock", self._backend)
            self._mock = True

    def _lexical_score(self, query: str, document_text: str) -> float:
        """Calculate a simple keyword overlap score (0.0 - 1.0)."""
        import re
        # Tokenise and clean
        q_tokens = set(re.findall(r'\w+', query.lower()))
        d_tokens = set(re.findall(r'\w+', document_text.lower()))
        if not q_tokens: return 0.0
        
        # Calculate overlap (percentage of query words found in document)
        matches = q_tokens.intersection(d_tokens)
        return len(matches) / len(q_tokens)

    # ── Public API ─────────────────────────────────────────────────────────────

    def query(
        self,
        question: str,
        product_type: Optional[str] = None,
        heritage_brand: Optional[str] = None,
        session_id: str = "",
        audit_logger=None,
    ) -> RAGResult:
        """
        Retrieve top-k FAQ chunks for the given question.
        Filters by product_type and heritage_brand where available.
        Returns RAGResult with answerable=False if no chunk meets the threshold.
        """
        query_hash = hashlib.sha256(
            f"{question.lower().strip()}:{product_type}".encode()
        ).hexdigest()[:16]
        cache_key  = f"rag:{query_hash}"

        import time
        start_t = time.perf_counter()

        # Cache check
        threshold = float(os.getenv("RAG_SCORE_THRESHOLD", "0.75"))
        cached = self._cache.get(cache_key)
        if cached:
            latency_ms = int((time.perf_counter() - start_t) * 1000)
            logger.info("[%s] RAG cache hit for query_hash=%s", session_id, query_hash)
            result = RAGResult(query=question, query_hash=query_hash,
                               chunks=cached, cache_hit=True)
            result.answerable = any(c.score >= threshold for c in cached)
            if audit_logger:
                audit_logger.log_rag_query(session_id, query_hash, product_type,
                                           [c.score for c in cached], hit=True,
                                           question=question, answerable=result.answerable,
                                           context="\n\n".join(c.text for c in cached), latency_ms=latency_ms)
            return result

        # Fetch
        if self._mock:
            chunks = self._mock_retrieve(question, product_type)
        elif self._backend == "chroma":
            chunks = self._chroma_retrieve(question, product_type, heritage_brand)
        elif self._backend == "pinecone":
            chunks = self._pinecone_retrieve(question, product_type, heritage_brand)
        else:
            chunks = self._pgvector_retrieve(question, product_type)

        self._cache.set(cache_key, chunks)
        answerable = any(c.score >= threshold for c in chunks)
        self._cache.set(query_hash, chunks)
        top_score = chunks[0].score if chunks else 0.0
        
        latency_ms = int((time.perf_counter() - start_t) * 1000)

        # Define answerable threshold
        is_answerable = top_score >= 0.55
        
        if audit_logger:
            audit_logger.log_rag_query(
                session_id, query_hash, product_type,
                [c.score for c in chunks], hit=False,
                question=question, answerable=is_answerable,
                context="\n\n".join(c.text for c in chunks), latency_ms=latency_ms
            )

        if not chunks:
            logger.warning("[%s] RAG: No chunks found for query: %r", session_id, question)
            return RAGResult(query=question, answerable=False, context="No relevant information found in knowledge base.")

        return RAGResult(
            query=question,
            chunks=chunks[:3],
            answerable=is_answerable,
            confidence=top_score
        )

    # ── Mock retrieval ─────────────────────────────────────────────────────────

    def _mock_retrieve(self, question: str, product_type: Optional[str]) -> list[RAGChunk]:
        all_faqs = self.list_all_faqs()
        
        # STRICT FILTER: Only show chunks that match the product_type
        candidates = []
        for c in all_faqs:
            c_type = c.get("product_type")
            if not product_type or c_type == "all" or c_type == product_type:
                candidates.append(c)
        scored = []
        for c in candidates:
            score = _mock_score(question, c["text"])
            scored.append(RAGChunk(
                chunk_id=c["chunk_id"], text=c["text"], score=score,
                source_doc=c["source_doc"], section=c["section"],
                product_type=c.get("product_type"),
            ))
        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:RAG_TOP_K]

    # ── Chroma retrieval ───────────────────────────────────────────────────────

    def _chroma_retrieve(self, question: str, product_type: Optional[str],
                         heritage_brand: Optional[str]) -> list[RAGChunk]:
        where = {}
        if product_type:
            where["product_type"] = {"$in": [product_type, "all"]}
            
        # 1. Fetch EVERYTHING to evaluate locally for Hybrid/BM25
        all_data = self._collection.get(where=where if where else None)
        if not all_data or not all_data.get("ids"):
            return []
            
        doc_ids = all_data["ids"]
        docs = all_data["documents"]
        metas = all_data["metadatas"]
        
        # 2. Dense Search (Chroma AI Vectors) - fetch distances for ALL items
        dense_results = self._collection.query(
            query_texts=[question],
            n_results=len(doc_ids),
            where=where if where else None,
        )
        if not dense_results.get("ids") or not dense_results["ids"][0]:
            return []
        
        dense_ranked_ids = dense_results["ids"][0]
        dense_distances = dense_results["distances"][0]
        
        # 3. Sparse Search (BM25 Exact Keywords)
        sparse_ranked_ids = []
        bm25_scores = []
        try:
            from rank_bm25 import BM25Okapi
            import numpy as np
            tokenized_docs = [str(doc).lower().split() for doc in docs]
            bm25 = BM25Okapi(tokenized_docs)
            tokenized_query = question.lower().split()
            bm25_scores = bm25.get_scores(tokenized_query)
            # Sort IDs by highest BM25 score
            sparse_ranked_ids = [doc_ids[i] for i in np.argsort(bm25_scores)[::-1]]
        except ImportError:
            logger.warning("rank_bm25 not installed — skipping keyword hybrid search")
            sparse_ranked_ids = dense_ranked_ids # Fallback

        # 4. Integrate via Reciprocal Rank Fusion (RRF)
        k = 60
        rrf_scores = {}
        for rank, doc_id in enumerate(dense_ranked_ids):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k + rank))
        for rank, doc_id in enumerate(sparse_ranked_ids):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k + rank))
            
        # Select top K chunk IDs purely based on the combined RRF ranking
        final_ranked_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:RAG_TOP_K]
        
        # 5. Pack RAGChunks, mapping confidence back to a 0.0-1.0 scale
        chunks = []
        for cid in final_ranked_ids:
            idx = doc_ids.index(cid)
            dense_idx = dense_ranked_ids.index(cid)
            raw_dense_score = 1.0 - float(dense_distances[dense_idx])
            
            # Apply BM25 Boosting: If the keyword explicitly matched, boost the semantic score
            hybrid_score = raw_dense_score
            if len(bm25_scores) > idx and bm25_scores[idx] > 0.5:
                # Add up to 0.25 bonus for a strong lexical match
                hybrid_score = min(1.0, raw_dense_score + 0.25)

            chunks.append(RAGChunk(
                chunk_id=cid, text=docs[idx], score=hybrid_score,
                source_doc=metas[idx].get("source_doc",""), section=metas[idx].get("section",""),
                product_type=metas[idx].get("product_type"),
            ))
            
        return chunks

    # ── pgvector retrieval ─────────────────────────────────────────────────────

    def _pgvector_retrieve(self, question: str, product_type: Optional[str]) -> list[RAGChunk]:
        # Embed the query first using cached model
        embedding = self._get_embedger().encode(question).tolist()

        with self._pg_conn.cursor() as cur:
            filter_clause = "WHERE product_type = %s OR product_type IS NULL" if product_type else ""
            params = [str(embedding), RAG_TOP_K]
            if product_type:
                params = [product_type, str(embedding), RAG_TOP_K]
            cur.execute(f"""
                SELECT chunk_id, content, 1 - (embedding <=> %s::vector) AS score,
                       source_doc, section, product_type
                FROM {COLLECTION}
                {filter_clause}
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """, ([product_type] if product_type else []) + [str(embedding), str(embedding), RAG_TOP_K])
            rows = cur.fetchall()
        return [
            RAGChunk(chunk_id=r[0], text=r[1], score=float(r[2]),
                     source_doc=r[3], section=r[4], product_type=r[5])
            for r in rows
        ]
    # ── Pinecone retrieval ─────────────────────────────────────────────────────
    
    def _pinecone_retrieve(self, question: str, product_type: Optional[str],
                           heritage_brand: Optional[str]) -> list[RAGChunk]:
        """Hybrid search (Semantic + Keyword) for Pinecone."""
        # 1. Embed query locally for semantic search using cached model
        query_vec = self._get_embedger().encode(question).tolist()
        
        # 2. Build filter
        filter_dict = {}
        if product_type:
            filter_dict["product_type"] = {"$in": [product_type, "all"]}
        if heritage_brand:
            filter_dict["heritage_brand"] = {"$in": [heritage_brand, "all"]}
            
        # 3. Query Pinecone (fetch more candidates for hybrid re-ranking)
        try:
            results = self._index.query(
                vector=query_vec,
                top_k=min(20, RAG_TOP_K * 4), 
                include_metadata=True,
                filter=filter_dict if filter_dict else None
            )
            
            hybrid_scored = []
            for match in results.get("matches", []):
                meta = match.get("metadata", {})
                chunk_text = meta.get("text", "")
                
                # Semantic Score from Pinecone
                semantic_score = match["score"]
                
                # Lexical (Keyword) Score locally
                lexical_score = self._lexical_score(question, chunk_text)
                
                # Combine: Final Score = Semantic + (Lexical Boost)
                # We give lexical overlap a significant weighting to ensure exact matches float to top
                final_score = min(1.0, semantic_score + (lexical_score * 0.30))
                
                hybrid_scored.append(RAGChunk(
                    chunk_id=match["id"],
                    text=chunk_text,
                    score=final_score,
                    source_doc=meta.get("source_doc", "unknown"),
                    section=meta.get("section", "unknown"),
                    product_type=meta.get("product_type"),
                    heritage_brand=meta.get("heritage_brand")
                ))
            
            # Sort by boosted hybrid score
            hybrid_scored.sort(key=lambda x: x.score, reverse=True)
            return hybrid_scored[:RAG_TOP_K]
        except Exception as e:
            logger.error("Pinecone query failed: %s", e)
            return []
