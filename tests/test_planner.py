"""Planner: question -> JSON plan -> SQL written by code. Uses the real Bluepeak sample Excel."""
import json
from pathlib import Path

import pytest

import data_agent
from data_agent import DataStore
from planner import Plan, PlanError, build_sql, parse_plan, validate_plan

SAMPLE = Path(__file__).resolve().parents[1] / "sample_files" / "Bluepeak_Sales_2026.xlsx"
SALES = "bluepeak_sales_2026_sales"


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    if not SAMPLE.exists():
        pytest.skip("sample_files/Bluepeak_Sales_2026.xlsx not found")
    s = DataStore(tmp_path_factory.mktemp("t"))
    s.add("bp", SAMPLE.name, SAMPLE.read_bytes())
    return s


def plan(**kw):
    base = {"table": SALES, "metric": "count_rows", "column": None, "group_by": None, "filters": [],
            "date_column": None, "date_from": None, "date_to": None, "order": None, "limit": None,
            "complex": False}
    return {**base, **kw}


def run(store, raw):
    _, tables = store.connection(["bp"])
    p, t = validate_plan(raw, tables)
    return store.run_sql(build_sql(p, t), ["bp"])[1]


# Answer key from the sample files (checked earlier against the raw Excel).
@pytest.mark.parametrize("raw, expected", [
    (plan(metric="count_distinct", column="salesperson"), [[11]]),                 # the 563 bug
    (plan(metric="count_rows"), [[563]]),
    (plan(metric="sum", column="units", group_by="category", order="desc", limit=1), [["Phones", 568]]),
    (plan(metric="sum", column="units * unit_price * (1 - discount_pct / 100)", group_by="salesperson",
          order="desc", limit=1), [["Kabir Shah", 6359250.0]]),
    (plan(metric="sum", column="units", group_by="region", date_column="order_date",
          date_from="2026-07-01", date_to="2026-09-30"),
     [["East", 143], ["North", 204], ["South", 180], ["West", 172]]),                # Q3, whole last day
    (plan(metric="count_rows", filters=[{"column": "discount_pct", "op": ">", "value": 20}]), [[4]]),
    (plan(metric="count_rows", filters=[{"column": "region", "op": "=", "value": "north"}]), None),  # any case
])
def test_plans_give_answer_key_numbers(store, raw, expected):
    rows = run(store, raw)
    if expected is None:
        assert rows[0][0] > 0
    else:
        assert rows == expected


@pytest.mark.parametrize("raw, msg", [
    (plan(metric="count_distinct", column="sales_person"), "Unknown column"),
    (plan(metric="sum", column="region"), "Not a number column"),
    (plan(metric="sum", column="units; DROP TABLE x"), "Unsupported math"),
    (plan(metric="median", column="units"), "Unknown metric"),
    (plan(table="nope"), "Unknown table"),
    (plan(filters=[{"column": "units", "op": "LIKE", "value": 1}]), "Unknown operator"),
    (plan(date_column="order_date", date_from="July 1"), "Bad date"),
    (plan(complex=True), "free-form SQL"),
])
def test_bad_plans_are_rejected(store, raw, msg):
    _, tables = store.connection(["bp"])
    with pytest.raises(PlanError, match=msg):
        validate_plan(raw, tables)


def test_text_values_are_safely_quoted(store):
    rows = run(store, plan(metric="count_rows", filters=[{"column": "salesperson", "op": "=", "value": "O'Brien"}]))
    assert rows == [[0]]


def test_parse_plan_from_chatty_reply():
    assert parse_plan('Sure!\n```json\n{"metric": "count_rows"}\n```')["metric"] == "count_rows"
    with pytest.raises(PlanError):
        parse_plan("I think you want the number of salespeople.")


def test_describe():
    p = Plan(table="t", metric="count_distinct", column="salesperson")
    assert p.describe() == "number of different salesperson"


def test_ask_uses_plan_first(store, monkeypatch):
    calls = []

    def fake(system, user, model=None):
        calls.append(system[:20])
        return json.dumps(plan(metric="count_distinct", column="salesperson"))
    monkeypatch.setattr(data_agent, "_llm", fake)
    res = store.ask("how many sales persons are there", ["bp"])
    assert res.rows == [[11]] and 'COUNT(DISTINCT "salesperson")' in res.sql
    assert res.plan == "number of different salesperson" and len(calls) == 1   # no SQL-writing call needed


def test_ask_falls_back_to_free_sql_when_plan_is_bad(store, monkeypatch):
    replies = iter([json.dumps(plan(metric="sum", column="no_such_column")),
                    f'SELECT COUNT(DISTINCT "region") AS n FROM "{SALES}"'])
    monkeypatch.setattr(data_agent, "_llm", lambda s, u, model=None: next(replies))
    res = store.ask("how many regions", ["bp"])
    assert res.rows == [[4]] and res.plan == ""
