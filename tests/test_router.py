import pytest

import router
from router import route, rule_guess


@pytest.mark.parametrize("q, expected", [
    ("What is the leave policy?", "docs"),
    ("Top 5 customers by revenue", "data"),
    ("Total units by region", "data"),
    ("Did total sales meet the target in the plan?", "both"),
])
def test_rule_guess(q, expected):
    assert rule_guess(q) == expected


def test_no_llm_call_when_only_one_kind_selected(monkeypatch):
    def boom(**kw):
        raise AssertionError("LLM should not be called")
    monkeypatch.setattr(router.ollama, "chat", boom)
    assert route("anything", ["hr.pdf"], []).name == "docs"
    assert route("anything", [], ["sales.xlsx/Sales (units)"]).name == "data"


@pytest.mark.parametrize("reply, expected", [
    ("data", "data"), ("Both.", "both"), ("**docs** - it's in the policy", "docs"),
])
def test_llm_choice_is_parsed(monkeypatch, reply, expected):
    monkeypatch.setattr(router.ollama, "chat", lambda **kw: {"message": {"content": reply}})
    r = route("q", ["hr.pdf"], ["sales.xlsx/Sales (units)"])
    assert r.name == expected and r.reason == "chosen by the model"


def test_falls_back_to_rules_when_llm_unclear_or_down(monkeypatch):
    monkeypatch.setattr(router.ollama, "chat", lambda **kw: {"message": {"content": "hmm, not sure"}})
    assert route("Total units by region", ["hr.pdf"], ["s"]).name == "data"

    def down(**kw):
        raise ConnectionError
    monkeypatch.setattr(router.ollama, "chat", down)
    r = route("What is the leave policy?", ["hr.pdf"], ["s"])
    assert r.name == "docs" and "keyword" in r.reason


# ---------- regression: real questions that were mis-routed by llama3.2 ----------
@pytest.mark.parametrize("q, expected", [
    ("How many annual leave days do I get?", "docs"),            # was routed to data
    ("How many sick days am I allowed?", "docs"),
    ("Which form do I use to apply for leave?", "docs"),
    ("Which regions met their Q3 target?", "both"),              # was routed to data
    ("Which orders break the discount policy?", "both"),
    ("Which employees took more leave than the policy allows?", "both"),
    ("Which expenses exceeded the budget?", "both"),
])
def test_strong_patterns_decide_without_llm(monkeypatch, q, expected):
    def boom(**kw):
        raise AssertionError("LLM should not be called for clear patterns")
    monkeypatch.setattr(router.ollama, "chat", boom)
    assert route(q, ["handbook.pdf"], ["sales.xlsx/Sales (region, units)"]).name == expected


def test_router_prompt_includes_document_snippets(monkeypatch):
    seen = {}

    def fake(**kw):
        seen["system"] = kw["messages"][0]["content"]
        return {"message": {"content": "docs"}}
    monkeypatch.setattr(router.ollama, "chat", fake)
    route("What is the warranty on laptops?", ["h.pdf"], ["s.xlsx/Sales"],
          doc_snippets=["Laptops and phones come with a 24-month warranty."])
    assert "24-month warranty" in seen["system"]


def test_unclear_follow_up_keeps_previous_route(monkeypatch):
    monkeypatch.setattr(router.ollama, "chat", lambda **kw: {"message": {"content": "not sure"}})
    r = route("What are the names of the sales persons in the South region?", ["h.pdf"], ["s.xlsx/Employees"],
              previous_route="data")
    assert r.name == "data" and r.reason == "follow-up: same source as the previous answer"
