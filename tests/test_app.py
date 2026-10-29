"""Runs the real Streamlit app through all three routes with a fake Ollama."""
import types
from pathlib import Path

import ollama
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

import rag
from data_agent import DataStore
from rag import Chunk, PdfStore
from tests.test_data_agent import sales_xlsx

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def fake_chat(model, messages, stream=False, options=None):
    system, user = messages[0]["content"], messages[-1]["content"]
    if system.startswith("You route"):
        out = "both" if "target" in user.lower() else "docs" if "policy" in user.lower() else "data"
    elif system.startswith("You write ONE DuckDB"):
        if "target" in user.lower():
            assert "target for North is 20 units" in system      # "both" passes PDF facts to SQL
        out = ('```sql\nSELECT "region", sum("units") AS units FROM "sales_sales" '
               'WHERE "region" ILIKE \'north\' GROUP BY 1;\n```')
    elif system.startswith("You explain"):
        out = "North sold 15 units [sales.xlsx, SQL]."
    elif system.startswith("Answer using BOTH"):
        out = "North sold 15 units [sales.xlsx, SQL] vs a target of 20 [plan.pdf, p. 2], so it missed."
    elif system.startswith("You rewrite"):
        out = user.split("Follow-up: ")[1].split("\n")[0]
    else:  # docs answer
        out = "Staff get 20 days of leave [plan.pdf, p. 1]."
    if stream:
        return iter([{"message": {"content": w + " "}} for w in out.split(" ")])
    return {"message": {"content": out}}


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)                      # app stores data under ./chroma_db here
    monkeypatch.setattr(ollama, "chat", fake_chat)
    monkeypatch.setattr(ollama, "list", lambda: types.SimpleNamespace(models=[
        types.SimpleNamespace(model="llama3.2:latest"), types.SimpleNamespace(model="nomic-embed-text:latest")]))
    monkeypatch.setattr(rag, "embed", lambda texts, kind: [[t.lower().count("target"), 1.0] for t in texts])
    PdfStore().add("d1", "plan.pdf", [Chunk("Leave policy: staff get 20 days.", 1),
                                      Chunk("The Q3 target for North is 20 units.", 2)])
    DataStore().add("x1", "sales.xlsx", sales_xlsx())
    st.cache_resource.clear()
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception, at.exception
    return at


def ask(at, q):
    at.chat_input[0].set_value(q).run()
    assert not at.exception, at.exception
    return at.session_state.messages[-1]


def test_all_three_routes(app):
    data = ask(app, "Total units for North?")
    assert data["route"] == "data"
    assert data["data"]["rows"] == [["North", 15]] and data["data"]["error"] is None
    assert "15 units" in data["content"]

    both = ask(app, "Did North meet the Q3 target?")
    assert both["route"] == "both"
    assert both["data"]["rows"] == [["North", 15]]
    assert both["sources"] and "missed" in both["content"]

    docs = ask(app, "What is the leave policy?")
    assert docs["route"] == "docs" and docs["data"] is None
    assert "20 days" in docs["content"]

    # The SQL is shown to the user, results render as a plain HTML table (works in Safari too),
    # and the route reason is kept when old messages are redrawn.
    assert any("sales_sales" in c.value for c in app.code)
    # Actual vs target is calculated by code: North sold 15, the plan says 20 -> missed.
    comp = both["comparison"]
    assert comp == [{"group": "North", "actual": "15", "target": "20", "difference": "-5", "status": "missed"}]
    assert any("❌ missed" in t.value.to_string() for t in app.table)
    assert any(t.value.iloc[0].tolist() == ["North", 15] for t in app.table)
    assert both["reason"] == "compares data with a target, limit or policy"
    labels = app.sidebar.multiselect[0].options
    assert any("📊" in o for o in labels) and any("📄" in o for o in labels)
