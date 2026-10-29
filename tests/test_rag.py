"""Unit tests for the parts that don't need Ollama running. Run: pytest -q"""
import pytest

import rag
from rag import Chunk, PdfStore, bm25_scores, build_messages, chunk_pages, file_hash, rrf_fuse, tokenize
import numpy as np


@pytest.fixture
def fake_embed(monkeypatch):
    """Replace Ollama with a tiny fake embedder: counts 'cat' and 'dog' words."""
    def _embed(texts, kind):
        return [[t.count("cat"), t.count("dog"), 0.1] for t in texts]
    monkeypatch.setattr(rag, "embed", _embed)


def test_store_add_search_and_reload(tmp_path, fake_embed):
    store = PdfStore(str(tmp_path))
    assert not store.has("doc1")
    store.add("doc1", "x.pdf", [Chunk("cat cat cat", 1), Chunk("dog dog", 2)])
    assert store.has("doc1")

    hits = PdfStore(str(tmp_path)).search("cat", ["doc1"], k=5)  # fresh store = loads from disk
    assert [h["page"] for h in hits] == [1, 2]  # best match first, k capped at chunk count
    assert hits[0]["score"] > hits[1]["score"]


def test_store_rejects_empty(tmp_path, fake_embed):
    with pytest.raises(ValueError):
        PdfStore(str(tmp_path)).add("doc1", "x.pdf", [])


def test_chunks_keep_page_numbers():
    chunks = chunk_pages([(1, "a" * 100), (2, "b" * 100)], size=50, overlap=10)
    assert {c.page for c in chunks} == {1, 2}
    assert all(len(c.text) <= 50 for c in chunks)


def test_overlap_repeats_text_between_chunks():
    text = "".join(str(i % 10) for i in range(200))
    chunks = chunk_pages([(1, text)], size=100, overlap=20)
    assert chunks[0].text[-20:] == chunks[1].text[:20]


def test_short_page_gives_one_chunk():
    assert len(chunk_pages([(1, "hello world")], size=100, overlap=10)) == 1


def test_whitespace_is_cleaned_and_empty_pages_skipped():
    chunks = chunk_pages([(1, "hello\n\n   world"), (2, "   ")], size=100, overlap=10)
    assert [c.text for c in chunks] == ["hello world"]


@pytest.mark.parametrize("size,overlap", [(0, 0), (50, 50), (50, -1)])
def test_bad_settings_raise(size, overlap):
    with pytest.raises(ValueError):
        chunk_pages([(1, "x")], size=size, overlap=overlap)


def test_file_hash_is_stable():
    assert file_hash(b"abc") == file_hash(b"abc")
    assert file_hash(b"abc") != file_hash(b"abd")


def test_messages_include_context_and_limit_history():
    history = [{"role": "user", "content": str(i), "sources": []} for i in range(10)]
    msgs = build_messages("Q?", [{"text": "ctx", "page": 3, "source": "x.pdf"}], history)
    assert "[x.pdf, p. 3] ctx" in msgs[0]["content"]
    assert len(msgs) == 1 + 6 + 1  # system + last 6 turns + question
    assert "sources" not in msgs[1]  # UI-only fields are not sent to the model


def test_chunks_end_on_sentence_boundaries():
    text = "Leave policy applies to all staff. " * 30      # ~1050 chars, many sentences
    chunks = chunk_pages([(1, text)], size=200, overlap=50)
    assert len(chunks) > 1
    assert all(c.text.endswith(".") for c in chunks)      # never cut mid-sentence
    assert all(len(c.text) <= 200 for c in chunks)


def test_overlap_repeats_whole_last_sentence():
    text = "First sentence here. Second sentence here. Third sentence here. Fourth one."
    chunks = chunk_pages([(1, text)], size=45, overlap=25)
    assert chunks[0].text.endswith("Second sentence here.")
    assert chunks[1].text.startswith("Second sentence here.")


def test_paragraph_line_wraps_are_joined():
    text = "This line wraps\nonto the next line.\n\nNew paragraph."
    assert chunk_pages([(1, text)], size=200, overlap=20)[0].text == (
        "This line wraps onto the next line. New paragraph."
    )


def test_decimals_and_abbrev_like_numbers_not_split():
    text = "Price rose 2.5 percent in Q1. It fell later."
    from rag import split_sentences
    assert split_sentences(text) == ["Price rose 2.5 percent in Q1.", "It fell later."]


def test_tokenize_keeps_codes_and_drops_stopwords():
    assert tokenize("What is Section 4.2 of the HR-Policy?") == ["section", "4.2", "hr-policy"]


def test_bm25_prefers_exact_rare_terms():
    docs = [tokenize("leave policy for staff"), tokenize("leave policy code LP-77 applies"),
            tokenize("canteen menu")]
    scores = bm25_scores(tokenize("LP-77 leave"), docs)
    assert scores.argmax() == 1          # chunk with the exact code wins
    assert scores[2] == 0                # no shared words = zero
    assert bm25_scores([], docs).sum() == 0


def test_rrf_rewards_agreement_and_ignores_non_matches():
    vec = np.array([0.9, 0.8, 0.1])
    kw = np.array([0.0, 5.0, 0.0])        # only item 1 has the keyword
    fused = rrf_fuse(vec, kw)
    assert fused.argmax() == 1            # good in both lists beats best in one
    assert fused[2] < fused[0]


def test_rrf_ties_share_rank_so_keywords_decide():
    vec = np.array([0.5, 0.5])            # vectors can't tell them apart
    kw = np.array([0.0, 2.0])
    assert rrf_fuse(vec, kw).argmax() == 1


def test_hybrid_search_finds_exact_code(tmp_path, fake_embed):
    store = PdfStore(str(tmp_path))
    # Both chunks look identical to the (fake) embedder: one "cat" each.
    store.add("d", "policy.pdf", [Chunk("cat policy A7 overview", 1), Chunk("cat policy B9 details", 2)])
    hits = store.search("cat B9", ["d"], k=2)
    assert hits[0]["page"] == 2 and hits[0]["keyword"] > 0


def test_search_across_docs_and_filter(tmp_path, fake_embed):
    store = PdfStore(str(tmp_path))
    store.add("a", "cats.pdf", [Chunk("cat cat", 1)])
    store.add("b", "dogs.pdf", [Chunk("dog dog", 4)])
    hits = store.search("dog", ["a", "b"], k=2)
    assert hits[0]["source"] == "dogs.pdf" and hits[0]["page"] == 4
    assert {h["source"] for h in store.search("dog", ["a"], k=2)} == {"cats.pdf"}
    assert store.search("dog", []) == []
    assert store.list_docs() == {"a": "cats.pdf", "b": "dogs.pdf"}


def test_rewrite_skipped_without_history(monkeypatch):
    def boom(**kw):
        raise AssertionError("LLM should not be called")
    monkeypatch.setattr(rag.ollama, "chat", boom)
    assert rag.rewrite_question("What is a venv?", []) == "What is a venv?"


def test_rewrite_uses_model_output(monkeypatch):
    monkeypatch.setattr(rag.ollama, "chat",
        lambda **kw: {"message": {"content": '"Give an example of Python virtual environments"\n'}})
    history = [{"role": "user", "content": "what is environments"},
               {"role": "assistant", "content": "Isolated Python environments..."}]
    assert rag.rewrite_question("explain with an example", history) == \
        "Give an example of Python virtual environments"


@pytest.mark.parametrize("reply", ["", "x" * 500])
def test_rewrite_falls_back_when_model_fails_or_rambles(monkeypatch, reply):
    monkeypatch.setattr(rag.ollama, "chat", lambda **kw: {"message": {"content": reply}})
    history = [{"role": "user", "content": "what is environments"}]
    assert rag.rewrite_question("give an example", history) == "what is environments give an example"


def test_rewrite_falls_back_when_ollama_errors(monkeypatch):
    def down(**kw):
        raise ConnectionError("ollama not running")
    monkeypatch.setattr(rag.ollama, "chat", down)
    history = [{"role": "user", "content": "what is environments"}]
    assert "environments" in rag.rewrite_question("more detail", history)
