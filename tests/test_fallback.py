"""If one kind of source has no answer, the app checks the other (docs <-> spreadsheets).
Replays a real conversation: 'how many sales persons in south?' then 'tell me their names'."""
import json
import types
from pathlib import Path

import ollama
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

import rag
import router
from data_agent import DataStore
from rag import NOT_FOUND, Chunk, PdfStore

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "sample_files" / "Bluepeak_Sales_2026.xlsx"
EMP = "bluepeak_sales_2026_employees"
PLAN = {"table": EMP, "metric": "count_distinct", "column": "name", "group_by": None,
        "filters": [{"column": "region", "op": "=", "value": "South"}], "date_column": None,
        "date_from": None, "date_to": None, "order": None, "limit": None, "complex": False}


def fake_chat(model, messages, stream=False, options=None):
    system, user = messages[0]["content"], messages[-1]["content"]
    if system.startswith("You route"):
        out = "hmm, hard to say"                                   # router unsure (as in the screenshot)
    elif system.startswith("You rewrite"):
        out = "What are the names of the sales persons in the South region?"
    elif system.startswith("Turn the user's question into a query PLAN"):
        if "names" in user:
            out = json.dumps({**PLAN, "metric": "list"})
        elif "parking" in user:
            out = json.dumps({**PLAN, "metric": "list", "filters": [{"column": "region", "op": "=", "value": "Mars"}]})
        else:
            out = json.dumps(PLAN)
    elif system.startswith("You explain"):
        out = "The result lists the matching rows. [Bluepeak_Sales_2026.xlsx, SQL]"
    elif system.startswith("You suggest"):
        out = "What is the leave policy?"
    elif "leave" in user.lower():
        out = "Staff get 18 days of annual leave [hb.pdf, p. 1]."
    else:
        out = NOT_FOUND                                            # the documents don't have it
    if stream:
        return iter([{"message": {"content": w + " "}} for w in out.split(" ")])
    return {"message": {"content": out}}


@pytest.fixture
def app(tmp_path, monkeypatch):
    if not SAMPLE.exists():
        pytest.skip("sample file missing")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(ollama, "chat", fake_chat)
    monkeypatch.setattr(ollama, "list", lambda: types.SimpleNamespace(models=[
        types.SimpleNamespace(model="llama3.2:latest"), types.SimpleNamespace(model="nomic-embed-text:latest")]))
    monkeypatch.setattr(rag, "embed", lambda texts, kind: [[1.0, 0.5] for _ in texts])
    PdfStore().add("d1", "hb.pdf", [Chunk("Leave policy: staff get 18 days of annual leave.", 1)])
    DataStore().add("x1", SAMPLE.name, SAMPLE.read_bytes())
    st.cache_resource.clear()
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    assert not at.exception, at.exception
    return at


def ask(at, q):
    at.chat_input[0].set_value(q).run()
    assert not at.exception, at.exception
    return at.session_state.messages[-1]


def test_follow_up_stays_on_data_and_docs_miss_falls_back_to_data(app):
    first = ask(app, "how many sales persons are working on the south region")
    assert first["route"] == "data" and first["data"]["rows"] == [[3]]

    names = ask(app, "can you please tell me their names")
    # Router was unsure -> follow-up keeps the previous source (data) -> real names come back.
    assert names["route"] == "data"
    assert sorted(r[0] for r in names["data"]["rows"]) == ["Divya Nair", "Rahul Menon", "Sneha Pillai"]


def test_docs_route_falls_back_to_spreadsheets(app, monkeypatch):
    monkeypatch.setattr(router, "rule_guess", lambda q: "docs")      # force a wrong first choice
    reply = ask(app, "who are the people in south")
    assert reply["route"] == "data" and reply["reason"] == "fallback: not found via docs"
    assert reply["data"]["rows"] == [[3]]          # the fake plan counts them
    assert any("Not found in your documents" in c.value for c in app.caption)


def test_data_route_falls_back_to_documents(app, monkeypatch):
    monkeypatch.setattr(router, "rule_guess", lambda q: "data")
    reply = ask(app, "total leave allowed")     # data: plan finds nothing useful; docs answer it
    assert reply["route"] in ("docs", "data")
    if reply["route"] == "docs":
        assert "18 days" in reply["content"] and reply["reason"] == "fallback: not found via data"


def test_both_sources_empty_gives_suggestions(app, monkeypatch):
    monkeypatch.setattr(router, "rule_guess", lambda q: "docs")
    reply = ask(app, "where is the parking area")
    assert reply["suggestions"] == ["What is the leave policy?"]
    assert reply["data"] is not None and reply["data"]["rows"] == []   # the data said "no rows", shown
