"""
Router: decides which tool answers a question.

    docs -> RAG over PDFs / Word / text        ("How many sick leaves do I get?")
    data -> SQL over spreadsheets              ("Top 5 customers by revenue?")
    both -> RAG + SQL combined                 ("Did Q3 sales meet the target in the plan?")

Cheap first: if only one kind of source is selected, no LLM call is needed.
When both kinds are selected, a short LLM call decides, with keyword rules as a fallback.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import ollama

from rag import CHAT_MODEL

ROUTES = ("docs", "data", "both")

# Words that usually mean "calculate something over rows".
_DATA_WORDS = re.compile(
    r"\b(total|sum|average|avg|mean|median|count|how many|number of|top|bottom|highest|lowest|"
    r"max(imum)?|min(imum)?|most|least|rank|per|group|trend|growth|percent(age)?|ratio|"
    r"by (month|week|day|year|quarter|region|category|product|customer)|list all|which rows|"
    r"show (me )?(the )?(order|row|record|employee|customer)s?|names of|their names|who are|list (the|all)?)\b", re.I)
# Words that usually point at written documents.
_DOC_WORDS = re.compile(
    r"\b(policy|policies|rule|rules|guideline|procedure|target|goal|plan|contract|agreement|"
    r"according to|document|pdf|says?|mention(ed|s)?|explain|what is|define|definition|why)\b", re.I)

# Strong patterns decided WITHOUT the LLM (small models get these wrong too often).
# "compare" words + "rule" words = the answer needs table values checked against a document.
_COMPARE = re.compile(r"\b(met|meet|meeting|beat|beats|miss(ed|es)?|exceed(s|ed)?|break(s)?|broke|"
                      r"violat\w*|(more|less|fewer)( \w+)? than|over|under|above|below|within)\b", re.I)
_RULE = re.compile(r"\b(targets?|quotas?|polic(y|ies)|limits?|allow(s|ed)?|allowance|rules?|budgets?|caps?)\b", re.I)
# Questions about what I'm entitled to / how to do something = written rules.
_ENTITLEMENT = re.compile(r"\b(do i get|am i allowed|can i|should i|how do i|which form|"
                          r"what is the (policy|process|procedure|rule)|who approves)\b", re.I)

ROUTER_PROMPT = """You route a user's question to the right source.

DOCUMENTS contain written rules, policies, targets and facts: {docs}
Most relevant document text:
{snippets}

TABLES contain records (rows of data): {tables}

Reply with ONE word: docs, data or both.
- docs: the answer is a rule, limit, allowance, definition, process or fact written in the documents.
  Examples: "How many sick days am I allowed?", "Who approves travel?", "What is the refund window?"
- data: the answer needs counting, summing, ranking or filtering table rows.
  Examples: "Total units by category", "Top 3 products by revenue", "Show order 123"
- both: the question checks table values against a rule, target or limit from the documents.
  Examples: "Which teams beat their quota?", "Which expenses exceeded the limit?"
"""


@dataclass
class Route:
    name: str      # "docs" | "data" | "both"
    reason: str    # shown in the UI so users see why


def rule_guess(question: str) -> str:
    """Keyword fallback when the LLM is unavailable or unclear."""
    data, docs = bool(_DATA_WORDS.search(question)), bool(_DOC_WORDS.search(question))
    if data and docs:
        return "both"
    return "data" if data else "docs"


def route(question: str, doc_names: list[str], table_info: list[str],
          doc_snippets: list[str] | None = None, previous_route: str | None = None) -> Route:
    """doc_names: selected text documents. table_info: e.g. ["sales.xlsx (region, units, revenue)"].
    doc_snippets: the best-matching document text, so the router knows what the documents contain.
    previous_route: route of the previous answer, for follow-ups like "tell me their names"."""
    if not table_info:
        return Route("docs", "only documents are selected")
    if not doc_names:
        return Route("data", "only spreadsheets are selected")
    if _COMPARE.search(question) and _RULE.search(question):
        return Route("both", "compares data with a target, limit or policy")
    # "How many leave days do I get?" is an entitlement, so ignore "how many" when checking for data words.
    if _ENTITLEMENT.search(question) and not _DATA_WORDS.search(re.sub(r"how many", "", question, flags=re.I)):
        return Route("docs", "asks about a rule or entitlement")
    snippets = "\n".join(f"- {s[:250]}" for s in (doc_snippets or [])[:3]) or "(none)"
    try:
        resp = ollama.chat(
            model=CHAT_MODEL,
            options={"temperature": 0},
            messages=[
                {"role": "system", "content": ROUTER_PROMPT.format(
                    docs=", ".join(doc_names), snippets=snippets, tables="; ".join(table_info))},
                {"role": "user", "content": question},
            ],
        )
        words = re.findall(r"[a-z]+", resp["message"]["content"].lower())
        choice = next((w for w in words if w in ROUTES), None)
    except Exception:
        choice = None
    if choice:
        return Route(choice, "chosen by the model")
    if previous_route in ROUTES:   # unclear follow-up: stay with the source of the previous answer
        return Route(previous_route, "follow-up: same source as the previous answer")
    return Route(rule_guess(question), "keyword rules (model gave no clear answer)")
