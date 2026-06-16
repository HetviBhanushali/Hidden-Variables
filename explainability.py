"""
explainability.py — Member 4: Explainability & Evaluation
Hidden Variables RAG Project

Provides:
  - Source Highlighting  (page + paragraph attribution)
  - Confidence Scoring   (retrieval similarity → human-readable %)
  - Answer Validation    (hallucination guard)
  - PDF Summarization    (executive summary, key topics, conclusions)
  - Evaluation Metrics   (retrieval accuracy, latency, relevance)
"""

import os
import time
import json
from dataclasses import dataclass, field, asdict
from typing import Optional
from datetime import datetime

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
from huggingface_hub import login

load_dotenv()

# ─── Auth ────────────────────────────────────────────────────────────────────
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("hf_token")
GROQ_API_KEY = os.getenv("API_KEY") or os.getenv("APIKEY")

if HF_TOKEN:
    login(token=HF_TOKEN)


# ─── Data classes ────────────────────────────────────────────────────────────

@dataclass
class SourceReference:
    """Granular citation for one retrieved chunk."""
    filename: str
    page: int
    paragraph: int          # chunk index within the page
    snippet: str            # first 120 chars of the chunk
    raw_score: float        # cosine distance (lower = more similar)
    confidence_pct: float   # human-readable 0-100


@dataclass
class ExplainedAnswer:
    """Full explainability payload returned to the UI."""
    question: str
    answer: str
    sources: list[SourceReference]
    top_confidence: float       # highest confidence among sources
    found_in_document: bool     # False → answer is "not found"
    latency_ms: float
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class PDFSummary:
    executive_summary: str
    key_topics: list[str]
    main_conclusions: list[str]
    page_count: int
    chunk_count: int


@dataclass
class EvalMetrics:
    """Accumulated evaluation counters — persisted to metrics_log.json."""
    total_queries: int = 0
    answered_queries: int = 0
    unanswered_queries: int = 0
    avg_latency_ms: float = 0.0
    avg_top_confidence: float = 0.0
    avg_relevance_score: float = 0.0   # mean cosine distance of top-1 hit


# ─── Helpers ─────────────────────────────────────────────────────────────────

METRICS_FILE = "metrics_log.json"

_CONF_NOT_FOUND_THRESHOLD = 0.55   # distance above this → treat as "not found"
_CONF_SCALE_LOW  = 0.30            # distance at which confidence = ~100 %
_CONF_SCALE_HIGH = 1.00            # distance at which confidence = ~0 %


def _distance_to_confidence(distance: float) -> float:
    """
    Convert a Chroma cosine distance [0,1] to a confidence percentage [0,100].
    Lower distance = more similar = higher confidence.
    """
    clamped = max(0.0, min(1.0, distance))
    confidence = 1.0 - clamped          # invert: 0 dist → 1.0 conf
    return round(confidence * 100, 1)


def _extract_paragraph_index(chunk_index: int, docs_on_page: list) -> int:
    """Simple paragraph index: position of this chunk among same-page chunks."""
    return chunk_index + 1


def _load_metrics() -> EvalMetrics:
    if os.path.exists(METRICS_FILE):
        try:
            with open(METRICS_FILE) as f:
                return EvalMetrics(**json.load(f))
        except Exception:
            pass
    return EvalMetrics()


def _save_metrics(m: EvalMetrics) -> None:
    with open(METRICS_FILE, "w") as f:
        json.dump(asdict(m), f, indent=2)


def _update_metrics(m: EvalMetrics, result: ExplainedAnswer) -> EvalMetrics:
    """Running-average update after each query."""
    n = m.total_queries
    m.total_queries += 1
    if result.found_in_document:
        m.answered_queries += 1
    else:
        m.unanswered_queries += 1

    # running averages
    m.avg_latency_ms = (m.avg_latency_ms * n + result.latency_ms) / (n + 1)
    m.avg_top_confidence = (
        m.avg_top_confidence * n + result.top_confidence
    ) / (n + 1)
    if result.sources:
        top_raw = result.sources[0].raw_score
        m.avg_relevance_score = (
            m.avg_relevance_score * n + top_raw
        ) / (n + 1)

    _save_metrics(m)
    return m


# ─── Core engine ─────────────────────────────────────────────────────────────

class ExplainabilityEngine:
    """
    Drop-in wrapper around the shared Chroma vector store.

    Usage
    -----
    engine = ExplainabilityEngine()
    result = engine.ask("What is CNN?")
    print(result.answer)
    print(result.sources[0])
    """

    def __init__(
        self,
        persist_directory: str = "./chroma_langchain_db",
        collection_name: str = "pdf-rag",
        embedding_model: str = "BAAI/bge-small-en-v1.5",
        top_k: int = 5,
    ):
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        self.vector_store = Chroma(
            collection_name=collection_name,
            persist_directory=persist_directory,
            embedding_function=self.embeddings,
        )
        self.llm = ChatGroq(
            model="llama-3.1-8b-instant",
            groq_api_key=GROQ_API_KEY,
        )
        self.top_k = top_k
        self.metrics = _load_metrics()

    # ── Source highlighting ──────────────────────────────────────────────────

    def _retrieve_with_sources(self, question: str) -> tuple[list, list]:
        """
        Returns (docs_with_scores, source_references).
        Uses similarity_search_with_score so we get raw distances.
        """
        raw_results = self.vector_store.similarity_search_with_score(
            question, k=self.top_k
        )

        # Group by page so we can assign paragraph index
        page_chunk_counter: dict[tuple, int] = {}
        source_refs: list[SourceReference] = []

        for doc, distance in raw_results:
            meta = doc.metadata
            filename = meta.get("source", meta.get("filename", "Unknown PDF"))
            filename = os.path.basename(filename)
            page = meta.get("page", 0) + 1   # 0-indexed → 1-indexed

            key = (filename, page)
            page_chunk_counter[key] = page_chunk_counter.get(key, 0) + 1
            paragraph = page_chunk_counter[key]

            confidence = _distance_to_confidence(distance)
            snippet = doc.page_content[:120].replace("\n", " ").strip()

            source_refs.append(SourceReference(
                filename=filename,
                page=page,
                paragraph=paragraph,
                snippet=snippet,
                raw_score=round(distance, 4),
                confidence_pct=confidence,
            ))

        return raw_results, source_refs

    # ── Answer validation ────────────────────────────────────────────────────

    def _is_answer_found(self, sources: list[SourceReference]) -> bool:
        """
        Declare 'not found' if the best retrieval score is too low.
        This is the primary hallucination guard — if nothing is close
        enough to the query, we don't pass it to the LLM at all.
        """
        if not sources:
            return False
        best_distance = sources[0].raw_score
        return best_distance < _CONF_NOT_FOUND_THRESHOLD

    # ── LLM answer generation ────────────────────────────────────────────────

    def _generate_answer(
        self, question: str, context_docs: list
    ) -> str:
        context_text = "\n\n---\n\n".join(
            [doc.page_content for doc, _ in context_docs]
        )
        prompt = f"""You are a precise document Q&A assistant.
Answer the question using ONLY the context below.
If the context does not contain enough information, say exactly:
"Answer not found in document."

Context:
{context_text}

Question: {question}

Answer:"""
        response = self.llm.invoke(prompt)
        return response.content.strip()

    # ── Public: ask ──────────────────────────────────────────────────────────

    def ask(self, question: str) -> ExplainedAnswer:
        """
        Main entry point.  Returns a fully explained answer with sources,
        confidence, and latency.
        """
        t0 = time.perf_counter()

        raw_results, sources = self._retrieve_with_sources(question)
        found = self._is_answer_found(sources)

        if not found:
            answer = "Answer not found in document."
        else:
            answer = self._generate_answer(question, raw_results)
            # Secondary validation: if the LLM says not found anyway, respect it
            if "answer not found" in answer.lower():
                found = False

        latency_ms = round((time.perf_counter() - t0) * 1000, 1)
        top_conf = sources[0].confidence_pct if sources else 0.0

        result = ExplainedAnswer(
            question=question,
            answer=answer,
            sources=sources if found else [],
            top_confidence=top_conf if found else 0.0,
            found_in_document=found,
            latency_ms=latency_ms,
        )

        self.metrics = _update_metrics(self.metrics, result)
        return result

    # ── Public: summarize ────────────────────────────────────────────────────

    def summarize_pdf(self, pdf_path: str) -> PDFSummary:
        """
        Generate an executive summary, key topics, and main conclusions
        for a given PDF using the LLM.
        """
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        page_count = len(pages)

        # Use first 8 pages for summarization (context limit)
        sample_pages = pages[:8]
        full_text = "\n\n".join(p.page_content for p in sample_pages)
        full_text = full_text[:6000]   # hard cap for LLM context

        prompt = f"""You are a document analyst.
Read the following PDF content and return a JSON object with exactly these keys:
  "executive_summary": a 3-5 sentence summary of the document
  "key_topics": a list of 5 key topics covered
  "main_conclusions": a list of 3-5 main conclusions or findings

Return ONLY valid JSON. No preamble, no markdown fences.

PDF Content:
{full_text}"""

        response = self.llm.invoke(prompt)
        raw = response.content.strip()

        try:
            data = json.loads(raw)
            return PDFSummary(
                executive_summary=data.get("executive_summary", ""),
                key_topics=data.get("key_topics", []),
                main_conclusions=data.get("main_conclusions", []),
                page_count=page_count,
                chunk_count=self.vector_store._collection.count(),
            )
        except json.JSONDecodeError:
            # Fallback: return raw text as summary
            return PDFSummary(
                executive_summary=raw[:500],
                key_topics=[],
                main_conclusions=[],
                page_count=page_count,
                chunk_count=self.vector_store._collection.count(),
            )

    # ── Public: metrics report ───────────────────────────────────────────────

    def get_metrics(self) -> dict:
        """Return current evaluation metrics as a dict."""
        m = self.metrics
        answer_rate = (
            round(m.answered_queries / m.total_queries * 100, 1)
            if m.total_queries else 0.0
        )
        return {
            "total_queries": m.total_queries,
            "answered_queries": m.answered_queries,
            "unanswered_queries": m.unanswered_queries,
            "retrieval_accuracy_pct": answer_rate,
            "avg_latency_ms": round(m.avg_latency_ms, 1),
            "avg_confidence_pct": round(m.avg_top_confidence, 1),
            "avg_relevance_score": round(m.avg_relevance_score, 4),
        }

    def reset_metrics(self) -> None:
        """Reset evaluation counters (useful for fresh test runs)."""
        self.metrics = EvalMetrics()
        _save_metrics(self.metrics)
        print("Metrics reset.")
