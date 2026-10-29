import pytest
import rag

HITS = [{"text": "A venv isolates packages.", "page": 16, "source": "notes.pdf"},
        {"text": "Create it with python3 -m venv .venv", "page": 16, "source": "notes.pdf"}]
HISTORY = [{"role": "user", "content": "what is environments"},
           {"role": "assistant", "content": "Isolated Python setups [notes.pdf, p. 16]"}]


def fake_llm(monkeypatch, reply):
    monkeypatch.setattr(rag.ollama, "chat", lambda **kw: {"message": {"content": reply}})


def test_prompt_forbids_own_examples():
    assert "Never invent examples" in rag.SYSTEM_PROMPT
    assert rag.NOT_FOUND in rag.SYSTEM_PROMPT


@pytest.mark.parametrize("answer, expected", [
    ("I couldn't find that in the documents.", True),
    ("Sorry, I could not find this in the PDFs.", True),
    ("A venv isolates packages [notes.pdf, p. 16].", False),
])
def test_is_not_found(answer, expected):
    assert rag.is_not_found(answer) is expected


def test_rewrite_skipped_without_history(monkeypatch):
    def boom(**kw):
        raise AssertionError("LLM should not be called")
    monkeypatch.setattr(rag.ollama, "chat", boom)
    assert rag.rewrite_question("What is a venv?", []) == "What is a venv?"


def test_rewrite_uses_model_output(monkeypatch):
    fake_llm(monkeypatch, '"Explain Python environments with an example"\n')
    assert rag.rewrite_question("explain with an example", HISTORY) == \
        "Explain Python environments with an example"


@pytest.mark.parametrize("reply", ["", "x" * 500])
def test_rewrite_falls_back_when_model_fails_or_rambles(monkeypatch, reply):
    fake_llm(monkeypatch, reply)
    assert rag.rewrite_question("give an example", HISTORY) == "what is environments give an example"


def test_suggestions_are_cleaned(monkeypatch):
    fake_llm(monkeypatch, "Sure! Here are some:\n1. What is a venv used for?\n- How do you create a venv?\n"
                          "2) What is a venv used for?\n* Why?\n")
    assert rag.suggest_questions(HISTORY, HITS) == ["What is a venv used for?", "How do you create a venv?"]


def test_suggestions_capped_at_n(monkeypatch):
    fake_llm(monkeypatch, "\n".join(f"Question number {i} here?" for i in range(6)))
    assert len(rag.suggest_questions(HISTORY, HITS, n=3)) == 3


def test_suggestions_fallback_when_llm_down(monkeypatch):
    def down(**kw):
        raise ConnectionError("ollama not running")
    monkeypatch.setattr(rag.ollama, "chat", down)
    assert rag.suggest_questions(HISTORY, HITS) == ["What does notes.pdf say on page 16?"]


def test_no_hits_no_suggestions():
    assert rag.suggest_questions(HISTORY, []) == []
