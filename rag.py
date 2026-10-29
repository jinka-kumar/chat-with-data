"""
Core RAG logic: read PDF -> chunk -> embed -> store -> retrieve -> answer.

Why a separate file?
- The UI (app.py) stays thin, so this logic can be unit-tested
  and reused later (e.g. behind a FastAPI endpoint) without changes.

Everything runs locally and free:
- Ollama = runs the LLM + embedding model on your laptop (no API key, no cost)
- NumPy  = a tiny vector store saved to a local folder (no heavy DB to install)
"""
from __future__ import annotations

import hashlib
import re
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterator
from collections import Counter

import numpy as np
import ollama
from pypdf import PdfReader

# A sentence ends at . ! or ? followed by space and a capital letter, digit or quote.
# Simple on purpose: good enough for most English PDFs, no extra library needed.
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")


# --- Config -----------------------------------------------------------------
# Read from env vars so you can swap models without touching code.
CHAT_MODEL = os.getenv("CHAT_MODEL", "llama3.2")          # ~2 GB, runs on most laptops
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")  # small, fast, good quality
DB_PATH = os.getenv("DB_PATH", "./chroma_db")

CHUNK_SIZE = 800      # characters per chunk: big enough for meaning, small enough to be precise
CHUNK_OVERLAP = 150   # overlap so a sentence cut at a boundary still appears whole in one chunk
TOP_K = 4             # how many chunks we give the LLM as context
EMBED_BATCH = 32      # embed in batches: much faster than one call per chunk
HYBRID = os.getenv("HYBRID", "1") == "1"   # set HYBRID=0 to compare with vector-only
RRF_K = 60            # standard constant for Reciprocal Rank Fusion

NOT_FOUND = "I couldn't find that in the documents."

SYSTEM_PROMPT = f"""You answer questions using ONLY the context from the user's documents.
Rules:
- Use only facts and examples that appear in the context. Never invent examples,
  code or facts, and never use outside knowledge.
- If the context does not answer the question, reply with exactly: {NOT_FOUND}
- Keep answers short and clear. Use bullet points when listing things.
- Cite the file and location for each fact, like [policy.pdf, p. 3] or [sales.xlsx, Sheet1, rows 2-21].
"""

REWRITE_PROMPT = """You rewrite follow-up questions for a document search engine.
Given the chat history and a follow-up, write ONE standalone question that names the topic
from the history. Return only the question. If the follow-up is already standalone,
return it unchanged."""

SUGGEST_PROMPT = """You suggest follow-up questions for a document Q&A app.
Write {n} short questions that:
- can be answered ONLY from the context below (do not ask about anything else),
- relate to the topic of the conversation when possible.
Return one question per line, no numbering, no extra text.

Context:
{context}"""


@dataclass
class Chunk:
    text: str
    page: int


# --- Step 1: Read ------------------------------------------------------------
def file_hash(data: bytes) -> str:
    """Stable ID for a file. Same PDF uploaded twice = same ID = no re-processing."""
    return hashlib.sha256(data).hexdigest()[:16]


def read_pdf(file: BinaryIO) -> list[tuple[int, str]]:
    """Return [(page_number, text), ...]. Skips empty pages (e.g. scanned images)."""
    reader = PdfReader(file)
    pages: list[tuple[int, str]] = []
    for page_no, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append((page_no, text))
    return pages


def split_paragraphs(text: str) -> list[str]:
    """Blank lines separate paragraphs. Single newlines inside a paragraph are
    just PDF line wraps, so we join them with spaces."""
    return [" ".join(p.split()) for p in re.split(r"\n\s*\n", text) if p.strip()]

def split_sentences(paragraph: str) -> list[str]:
    return [s for s in _SENTENCE_END.split(paragraph) if s.strip()]

def _hard_split(text: str, size: int, overlap: int) -> list[str]:
    """Fallback for one giant 'sentence' (tables, code, lists without full stops)."""
    step = size - overlap
    return [text[i : i + size] for i in range(0, max(len(text) - overlap, 1), step)]



# --- Step 2: Chunk -----------------------------------------------------------
def chunk_pages(
    pages: list[tuple[int, str]],
    size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[Chunk]:
    """Sentence-aware chunking: pack whole sentences into chunks of up to `size`
    characters, and repeat the last sentence(s) (up to `overlap` chars) at the
    start of the next chunk. Chunking per page keeps exact page citations."""
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError("Need size > 0 and 0 <= overlap < size")

    chunks: list[Chunk] = []
    for page_no, text in pages:
        # 1. Break the page into units: whole sentences (or hard pieces if too long).
        units: list[str] = []
        for para in split_paragraphs(text):
            for sent in split_sentences(para):
                units.extend(_hard_split(sent, size, overlap) if len(sent) > size else [sent])

        # 2. Greedily pack units into chunks.
        current: list[str] = []
        length = 0
        for unit in units:
            if current and length + len(unit) > size:
                chunks.append(Chunk(" ".join(current), page_no))
                # Overlap: carry the last sentences that fit within `overlap` chars.
                carry: list[str] = []
                carry_len = 0
                for s in reversed(current):
                    if carry_len + len(s) + 1 > overlap:
                        break
                    carry.insert(0, s)
                    carry_len += len(s) + 1
                # If carry + next unit would still be too big, drop the carry.
                current, length = (carry, carry_len) if carry_len + len(unit) <= size else ([], 0)
            current.append(unit)
            length += len(unit) + 1  # +1 for the joining space
        if current:
            chunks.append(Chunk(" ".join(current), page_no))
    return chunks


# --- Step 3: Embed -----------------------------------------------------------
def embed(texts: list[str], kind: str) -> list[list[float]]:
    """Turn text into vectors. nomic-embed-text works best with a task prefix."""
    prefix = "search_query: " if kind == "query" else "search_document: "
    resp = ollama.embed(model=EMBED_MODEL, input=[prefix + t for t in texts])
    return resp["embeddings"]


# --- Step 3b: Keyword search (BM25) -----------------------------------------
# Keeps codes and numbers together: "4.2", "hr-policy", "q1" stay one token.
_TOKEN = re.compile(r"[a-z0-9]+(?:[.\-_][a-z0-9]+)*")
_STOPWORDS = frozenset(
    "a an the and or of to in on for is are was were be by with as at it this that "
    "from what how do does can i you my our me".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS]


def bm25_scores(query_tokens: list[str], docs_tokens: list[list[str]],
                k1: float = 1.5, b: float = 0.75) -> np.ndarray:
    """Classic BM25: rare words that appear often in a short chunk score highest.
    k1 = how fast repeated words stop adding score; b = how much long chunks are penalised."""
    n = len(docs_tokens)
    scores = np.zeros(n, dtype=np.float32)
    if n == 0 or not query_tokens:
        return scores
    lengths = np.array([len(d) for d in docs_tokens], dtype=np.float32)
    avg_len = float(lengths.mean()) or 1.0
    counts = [Counter(d) for d in docs_tokens]
    for term in set(query_tokens):
        tf = np.array([c.get(term, 0) for c in counts], dtype=np.float32)
        df = int(np.count_nonzero(tf))
        if df == 0:
            continue
        idf = np.log(1 + (n - df + 0.5) / (df + 0.5))   # rare term = high weight
        scores += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * lengths / avg_len))
    return scores


def rrf_fuse(*score_lists: np.ndarray, k: int = RRF_K) -> np.ndarray:
    """Reciprocal Rank Fusion: combine rankings, not raw scores, because cosine
    (0..1) and BM25 (0..any) are on different scales. Only matches (score > 0) count."""
    fused = np.zeros(len(score_lists[0]), dtype=np.float64)
    for scores in score_lists:
        # Dense rank: equal scores share a rank, so ties don't get an arbitrary winner.
        _, inverse = np.unique(-scores, return_inverse=True)
        ranks = inverse + 1
        fused += np.where(scores > 0, 1.0 / (k + ranks), 0.0)
    return fused

# --- Step 4 + 5: Store and Search -------------------------------------------
def _normalize(vectors: np.ndarray) -> np.ndarray:
    """Scale each vector to length 1, so a dot product = cosine similarity."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.clip(norms, 1e-12, None)  # clip avoids divide-by-zero


class PdfStore:
    """A tiny vector store: one .npy (vectors) + one .json (text, pages) per PDF.

    Why not a vector DB? For a few PDFs (thousands of chunks), a NumPy dot product
    is just as fast, needs no native builds, and shows exactly how vector search works.
    Swap in pgvector/Qdrant later when you have millions of chunks.
    """

    def __init__(self, path: str = DB_PATH):
        self.dir = Path(path)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, tuple[np.ndarray, list[dict]]] = {}  # avoid re-reading disk per question
        self._tokens: dict[str, list[list[str]]] = {}   # tokenised chunks, per PDF
    def _paths(self, doc_id: str) -> tuple[Path, Path]:
        return self.dir / f"{doc_id}.npy", self.dir / f"{doc_id}.json"

    def has(self, doc_id: str) -> bool:
        vec_path, meta_path = self._paths(doc_id)
        return vec_path.exists() and meta_path.exists()

    def add(self, doc_id: str, source: str, chunks: list[Chunk], on_progress=None,
            labels: dict[int, str] | None = None) -> None:
        if not chunks:
            raise ValueError("No chunks to store")
        vectors: list[list[float]] = []
        for b in range(0, len(chunks), EMBED_BATCH):
            batch = chunks[b : b + EMBED_BATCH]
            vectors.extend(embed([c.text for c in batch], kind="document"))
            if on_progress:
                on_progress(min(b + EMBED_BATCH, len(chunks)) / len(chunks))

        matrix = _normalize(np.asarray(vectors, dtype=np.float32))
        labels = labels or {}
        # "loc" = where to look: "p. 3" for PDFs, "Sheet1, rows 2-21" for Excel, etc.
        meta = [{"text": c.text, "page": c.page, "source": source,
                 "loc": labels.get(c.page, f"p. {c.page}")} for c in chunks]

        vec_path, meta_path = self._paths(doc_id)
        # Write the JSON first and the vectors last: has() needs both files,
        # so a crash halfway never leaves a "ready" but broken document.
        meta_path.write_text(json.dumps(meta), encoding="utf-8")
        np.save(vec_path, matrix)
        self._cache[doc_id] = (matrix, meta)

    def _load(self, doc_id: str) -> tuple[np.ndarray, list[dict]]:
        if doc_id not in self._cache:
            vec_path, meta_path = self._paths(doc_id)
            self._cache[doc_id] = (
                np.load(vec_path),
                json.loads(meta_path.read_text(encoding="utf-8")),
            )
        return self._cache[doc_id]

    def list_docs(self) -> dict[str, str]:
        "All Pdf already indexed on disk: {doc_id: file name}"

        docs = {}
        for meta_path in sorted(self.dir.glob("*.json")):
            doc_id = meta_path.stem
            if self.has(doc_id): #skip half-written docs
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                docs[doc_id] = meta[0]["source"] if meta else doc_id
        return docs


    def search(self, question: str, doc_ids: list[str], k: int = TOP_K) -> list[dict]:
        """Hybrid search across several PDFs: meaning (vectors) + exact words (BM25)."""
        if not doc_ids:
            return []

        matrices, metas, tokens = [], [], []
        for doc_id in doc_ids:
            matrix, meta = self._load(doc_id)
            matrices.append(matrix)
            metas.extend(meta)
            if doc_id not in self._tokens:                      # tokenise once per PDF
                self._tokens[doc_id] = [tokenize(m["text"]) for m in meta]
            tokens.extend(self._tokens[doc_id])

        all_vectors = np.vstack(matrices)
        query = _normalize(np.asarray(embed([question], kind="query"), dtype=np.float32))[0]
        vector_scores = all_vectors @ query                     # meaning
        keyword_scores = bm25_scores(tokenize(question), tokens)  # exact words

        ranking = rrf_fuse(vector_scores, keyword_scores) if HYBRID else vector_scores
        top = np.argsort(-ranking)[: min(k, len(metas))]
        return [
            {"text": metas[i]["text"],
             "page": metas[i]["page"], "source": metas[i]["source"],
             "loc": metas[i].get("loc", f"p. {metas[i]['page']}"),   # old files have no "loc"
             "score": round(float(vector_scores[i]), 3),
             "keyword": round(float(keyword_scores[i]), 2)}
            for i in top
        ]


# --- Step 6: Answer ----------------------------------------------------------
def cite(hit: dict) -> str:
    """'policy.pdf, p. 3' or 'sales.xlsx, Sheet1, rows 2-21'. Old data has no 'loc', so fall back to the page."""
    return f"{hit['source']}, {hit.get('loc') or 'p. ' + str(hit['page'])}"


def _ask_llm(system: str, user: str) -> str:
    """One short, non-streamed LLM call. temperature=0 so the same input gives the same output."""
    resp = ollama.chat(
        model=CHAT_MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        options={"temperature": 0},
    )
    return resp["message"]["content"].strip()


def rewrite_question(question: str, history: list[dict]) -> str:
    """Turn a follow-up like "explain with an example" into a full question that names
    the topic ("Explain Python environments with an example"), so search finds it.
    Only runs when there is chat history (saves one LLM call on the first question)."""
    if not history:
        return question
    recent = "\n".join(f"{m['role']}: {m['content'][:300]}" for m in history[-4:])
    try:
        out = _ask_llm(REWRITE_PROMPT, f"Chat history:\n{recent}\n\nFollow-up: {question}\n\nStandalone question:")
        rewritten = out.splitlines()[0].strip().strip('"')
    except Exception:
        rewritten = ""
    # Guard: small models sometimes fail or ramble. Fall back to a cheap, safe option.
    if not rewritten or len(rewritten) > 300:
        last_user = next((m["content"] for m in reversed(history) if m["role"] == "user"), "")
        return f"{last_user} {question}".strip()
    return rewritten


def is_not_found(answer: str) -> bool:
    """True when the model said it couldn't answer (small models vary the wording a bit)."""
    return "couldn't find" in answer.lower() or "could not find" in answer.lower()


def suggest_questions(history: list[dict], hits: list[dict], n: int = 3) -> list[str]:
    """Follow-up questions built from the closest PDF chunks, so every suggestion
    is answerable from the documents."""
    if not hits:
        return []
    context = "\n\n".join(f"[{cite(h)}] {h['text']}" for h in hits)
    recent = "\n".join(f"{m['role']}: {m['content'][:200]}" for m in history[-4:] if m["role"] == "user")
    try:
        out = _ask_llm(SUGGEST_PROMPT.format(n=n, context=context),
                       f"Conversation so far:\n{recent or '(none)'}\n\nSuggested questions:")
    except Exception:
        out = ""
    suggestions = []
    for line in out.splitlines():
        q = re.sub(r"^\s*(?:[-*\u2022]|\d+[.)])\s*", "", line).strip().strip('"')  # drop "1." / "-" bullets
        if q.endswith("?") and 8 <= len(q) <= 150 and q not in suggestions:
            suggestions.append(q)
    if not suggestions:  # model failed: offer safe questions about the closest pages
        suggestions = [f"What does {h['source']} say on page {h['page']}?" for h in hits]
    return list(dict.fromkeys(suggestions))[:n]


def build_messages(question: str, hits: list[dict], history: list[dict]) -> list[dict]:
    context = "\n\n".join(f"[{cite(h)}] {h['text']}" for h in hits)
    # Keep only the last few turns: saves context window and keeps answers fast.
    recent = [{"role": m["role"], "content": m["content"]} for m in history[-6:]]
    return [
        {"role": "system", "content": f"{SYSTEM_PROMPT}\nContext:\n{context}"},
        *recent,
        {"role": "user", "content": question},
    ]


def answer_stream(question: str, hits: list[dict], history: list[dict]) -> Iterator[str]:
    """Stream tokens so the user sees the answer type out live (feels much faster)."""
    stream = ollama.chat(
        model=CHAT_MODEL,
        messages=build_messages(question, hits, history),
        stream=True,
        options={"temperature": 0.1},  # low = factual, fewer made-up answers
    )
    for part in stream:
        yield part["message"]["content"]
