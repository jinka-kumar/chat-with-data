"""
Streamlit UI for "Chat with your documents and data".
Run:  streamlit run app.py

One chat box, three routes:
  📄 docs -> RAG over PDF / Word / text (answers with citations)
  📊 data -> SQL over Excel / CSV (exact numbers, SQL shown)
  🔀 both -> facts from documents + numbers from spreadsheets
"""
from pathlib import Path

import ollama
import streamlit as st

from data_agent import (SQL_MODEL, TABLE_EXTENSIONS, DataResult, DataStore, combined_answer,
                        compare_with_targets, explain_answer)
from loaders import SUPPORTED, read_file
from rag import (CHAT_MODEL, EMBED_MODEL, NOT_FOUND, PdfStore, answer_stream, chunk_pages, cite,
                 file_hash, is_not_found, rewrite_question, suggest_questions)
from router import route

UNCHECKED_NOTE = (":material/verified: Answer written from the data table: the AI's wording "
                  "mentioned numbers that are not in your data, so it was replaced.")
ROUTE_ICONS = {"docs": ":material/description:", "data": ":material/table_chart:", "both": ":material/merge:"}

st.set_page_config(page_title="Chat with your data", page_icon=":material/forum:", layout="centered")

# Light custom styling for a cleaner, premium feel.
st.markdown(
    """
    <style>
      .block-container {max-width: 860px; padding-top: 2rem;}
      .hero h1 {font-size: 2.1rem; margin-bottom: .2rem;
                background: linear-gradient(90deg,#6366f1,#06b6d4);
                -webkit-background-clip: text; -webkit-text-fill-color: transparent;}
      .hero p {color: #6b7280; margin-top: 0;}
      [data-testid="stChatMessage"] {border-radius: 14px; animation: fadeIn .35s ease;}
      @keyframes fadeIn {from {opacity:0; transform: translateY(6px);} to {opacity:1; transform:none;}}
    </style>
    <div class="hero"><h1>Chat with your documents & data</h1>
    <p>PDFs, Word, Excel and CSV. 100% free, private and offline on your own laptop.</p></div>
    """,
    unsafe_allow_html=True,
)


# --- Health check: fail early with a clear fix, not a stack trace -----------
@st.cache_resource
def get_stores() -> tuple[PdfStore, DataStore]:
    return PdfStore(), DataStore()


def check_ollama() -> str | None:
    try:
        names = {m.model for m in ollama.list().models}
    except Exception:
        return "Ollama is not running. Open the Ollama app (or run `ollama serve`) and refresh."
    installed = names | {n.split(":")[0] for n in names}       # "llama3.2" matches "llama3.2:latest"
    missing = [m for m in dict.fromkeys((CHAT_MODEL, EMBED_MODEL, SQL_MODEL)) if m not in installed]
    if missing:
        return "Missing models. Run: " + "  ".join(f"`ollama pull {m}`" for m in missing)
    return None


problem = check_ollama()
if problem:
    st.error(problem, icon=":material/error:")
    st.stop()

store, data_store = get_stores()
st.session_state.setdefault("messages", [])
st.session_state.setdefault("pending_question", None)   # set when a suggestion is clicked

# --- Sidebar: upload + library ------------------------------------------------
with st.sidebar:
    st.subheader("Your files")
    uploads = st.file_uploader("Upload files", type=SUPPORTED, accept_multiple_files=True,
                               help="Documents: PDF, Word, text, Markdown · Data: Excel (.xlsx), CSV")

    for uploaded in uploads or []:
        data = uploaded.getvalue()
        doc_id = file_hash(data)
        is_table = Path(uploaded.name).suffix.lower().lstrip(".") in TABLE_EXTENSIONS
        if (data_store if is_table else store).has(doc_id):
            continue  # already processed: skip, saves time

        with st.status(f"Reading {uploaded.name}...", expanded=True) as status:
            try:
                if is_table:
                    # Spreadsheets -> DuckDB (exact maths on all rows). No slow embedding needed.
                    tables = data_store.add(doc_id, uploaded.name, data)
                    for t in tables:
                        st.write(f"Sheet **{t.sheet}**: {t.rows:,} rows × {len(t.columns):,} columns")
                    status.update(label=f"{uploaded.name} ready", state="complete", expanded=False)
                    continue
                sections, labels = read_file(uploaded.name, data)
            except Exception as exc:   # corrupt or unsupported file: skip it, keep the others
                status.update(label=f"{uploaded.name}: {exc}", state="error")
                continue
            if not sections:
                status.update(label=f"{uploaded.name}: no text found (scanned or empty?)", state="error")
                continue

            chunks = chunk_pages(sections)
            st.write(f"{len(sections)} sections → {len(chunks)} chunks")
            bar = st.progress(0.0, text="Creating embeddings...")
            store.add(doc_id, uploaded.name, chunks, on_progress=lambda p: bar.progress(p), labels=labels)
            status.update(label=f"{uploaded.name} ready", state="complete", expanded=False)

    doc_library = store.list_docs()          # {doc_id: file name}
    table_library = data_store.list_files()  # {doc_id: file name}
    library = {**doc_library, **table_library}
    selected = st.multiselect(
        "Search in",
        options=list(library),
        default=list(library),
        format_func=lambda d: ("📊 " if d in table_library else "📄 ") + library[d],
    )

    if st.session_state.messages and st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.caption(f"Chat: `{CHAT_MODEL}` · SQL: `{SQL_MODEL}` · Embeddings: `{EMBED_MODEL}`")

selected_docs = [d for d in selected if d in doc_library]
selected_tables = [d for d in selected if d in table_library]


# --- Display helpers ---------------------------------------------------------------
def show_sources(sources: list[dict]) -> None:
    with st.expander(f"Sources ({len(sources)})"):
        for s in sources:
            st.markdown(f"**{cite(s)}** · meaning {s['score']:.0%} · keyword {s['keyword']}")
            st.caption(s["text"][:300] + ("..." if len(s["text"]) > 300 else ""))


def show_data(d: dict) -> None:
    """SQL + result table, so users can check the numbers themselves."""
    label = "Data used (SQL)" if not d.get("error") else "Data query failed"
    with st.expander(label, icon=":material/table_chart:"):
        if d.get("plan"):
            st.caption(f":material/psychology: Understood as: **{d['plan']}**")
        if d.get("sql"):
            st.code(d["sql"], language="sql")
        if d.get("error"):
            st.error(d["error"])
        elif d.get("rows"):
            # st.table = plain HTML: renders in every browser (st.dataframe's canvas grid
            # can show blank cells in some browsers). Results are capped at 50 rows anyway.
            st.table({c: [r[i] for r in d["rows"]] for i, c in enumerate(_unique(d["columns"]))})
            if d.get("truncated"):
                st.caption("Showing the first rows only.")
        else:
            st.caption("The query returned no rows.")


def show_comparison(rows: list[dict]) -> None:
    """Actual vs target, calculated by code (not by the AI), so it can be trusted."""
    icon = {"met": "✅ met", "missed": "❌ missed", "unknown": "❔ unknown"}
    st.markdown("**Actual vs target** (calculated exactly from your data and documents)")
    st.table({"Group": [r["group"] for r in rows], "Actual": [r["actual"] for r in rows],
              "Target": [r["target"] for r in rows], "Difference": [r["difference"] for r in rows],
              "Status": [icon[r["status"]] for r in rows]})


def _unique(cols: list[str]) -> list[str]:
    """SQL can return duplicate column names (e.g. two 'count'); tables need unique ones."""
    seen: dict[str, int] = {}
    out = []
    for c in cols:
        seen[c] = seen.get(c, 0) + 1
        out.append(c if seen[c] == 1 else f"{c}_{seen[c]}")
    return out


def ask(q: str) -> None:
    """Button callback: queue a suggested question as the next user message."""
    st.session_state.pending_question = q


def show_suggestions(suggestions: list[str], key: str) -> None:
    st.markdown("**Try one of these questions from your files:**")
    for j, q in enumerate(suggestions):
        st.button(q, key=f"{key}-{j}", on_click=ask, args=(q,), icon=":material/arrow_forward:")


def table_info(doc_ids: list[str]) -> list[str]:
    """Short description of each sheet for the router: 'sales.xlsx/Sales (region, units, ...)'."""
    out = []
    for d in doc_ids:
        for t in data_store.tables(d):
            cols = ", ".join(t.columns[:12]) + (", ..." if len(t.columns) > 12 else "")
            out.append(f"{t.source}/{t.sheet} ({cols})")
    return out


def data_payload(result: DataResult) -> dict:
    return {"sql": result.sql, "columns": result.columns, "rows": result.rows,
            "truncated": result.truncated, "error": result.error, "plan": result.plan}


# --- Chat -------------------------------------------------------------------
if not selected:
    st.info("Upload files or pick some in the sidebar.", icon=":material/upload_file:")

for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        if msg.get("route"):
            reason = f" ({msg['reason']})" if msg.get("reason") else ""
            st.caption(f"{ROUTE_ICONS[msg['route']]} Route: **{msg['route']}**{reason}")
        st.markdown(msg["content"])
        if msg.get("checked") is False:
            st.caption(UNCHECKED_NOTE)
        if msg.get("comparison"):
            show_comparison(msg["comparison"])
        if msg.get("data"):
            show_data(msg["data"])
        if msg.get("sources"):
            show_sources(msg["sources"])
        if msg.get("suggestions"):
            show_suggestions(msg["suggestions"], key=f"s{idx}")

typed = st.chat_input("Ask about your documents or data...", disabled=not selected)
question = typed or st.session_state.pending_question   # typed text, or a clicked suggestion
st.session_state.pending_question = None

if question:
    with st.chat_message("user"):
        st.markdown(question)

    history = st.session_state.messages
    reply = {"role": "assistant", "content": "", "sources": [], "suggestions": [], "data": None,
             "route": None, "reason": "", "comparison": None}

    with st.chat_message("assistant"):
        try:
            # 1. Follow-ups like "and by region?" -> a full question that names the topic.
            full_q = rewrite_question(question, history)
            if full_q != question:
                st.caption(f":material/search: Understood as: *{full_q}*")

            # 2. Search the documents first: the router uses the best matches to see what
            #    the documents contain, and the docs/both answers reuse the same hits.
            hits = store.search(full_q, selected_docs) if selected_docs else []

            # 3. Pick the route: docs, data or both.
            # A follow-up ("tell me their names") usually goes to the same source as the last answer.
            last = next((m for m in reversed(history) if m["role"] == "assistant" and m.get("route")), None)
            previous = last["route"] if (last and full_q != question) else None
            r = route(full_q, [doc_library[d] for d in selected_docs], table_info(selected_tables),
                      doc_snippets=[h["text"] for h in hits], previous_route=previous)
            reply["route"], reply["reason"] = r.name, r.reason
            st.caption(f"{ROUTE_ICONS[r.name]} Route: **{r.name}** ({r.reason})")

            if r.name == "both":   # give the SQL writer the document facts (targets, date ranges, limits)
                with st.spinner("Writing and running SQL on your data..."):
                    result = data_store.ask(full_q, selected_tables, facts=[h["text"] for h in hits])
                    comparison = compare_with_targets(full_q, hits, result)
                    answer, reply["checked"] = combined_answer(full_q, hits, result, comparison)
                st.markdown(answer)
                if reply["checked"] is False:
                    st.caption(UNCHECKED_NOTE)
                if comparison:
                    show_comparison(comparison)
                reply.update(data=data_payload(result), sources=hits, comparison=comparison)
                show_data(reply["data"])
                show_sources(hits)
            else:
                # Try the chosen source first. If it has no answer, try the OTHER kind of source
                # (documents <-> spreadsheets) before giving up.
                order = ["docs", "data"] if r.name == "docs" else ["data", "docs"]
                order = [s for s in order if (selected_docs if s == "docs" else selected_tables)]
                answer, used = None, None
                data_fallback = None            # keep a data result (e.g. "no rows") to show if all fail
                for i, source in enumerate(order):
                    if i > 0:
                        label = "documents" if order[0] == "docs" else "spreadsheets"
                        other = "spreadsheets" if source == "data" else "documents"
                        st.caption(f":material/swap_horiz: Not found in your {label}, so I checked your {other}.")
                    if source == "docs":
                        box = st.empty()       # stream into a box we can clear if it's "not found"
                        with box.container():
                            text = st.write_stream(answer_stream(question, hits, history))
                        if not is_not_found(text):
                            answer, used = text, "docs"
                            reply["sources"] = hits
                            show_sources(hits)
                            break
                        box.empty()
                    else:
                        with st.spinner("Writing and running SQL on your data..."):
                            result = data_store.ask(full_q, selected_tables)
                        if result.ok and result.rows:
                            with st.spinner("Explaining the result..."):
                                answer, reply["checked"] = explain_answer(result)
                            used = "data"
                            st.markdown(answer)
                            if reply["checked"] is False:
                                st.caption(UNCHECKED_NOTE)
                            reply["data"] = data_payload(result)
                            show_data(reply["data"])
                            break
                        data_fallback = result
                if used and used != r.name:
                    reply["route"], reply["reason"] = used, f"fallback: not found via {r.name}"
                if answer is None:
                    if data_fallback is not None:          # show what the data query said (e.g. no rows)
                        answer, reply["checked"] = explain_answer(data_fallback)
                        st.markdown(answer)
                        reply["data"] = data_payload(data_fallback)
                        show_data(reply["data"])
                    else:
                        answer = NOT_FOUND
                        st.markdown(answer)
                    if selected_docs:                       # suggest questions the files CAN answer
                        with st.spinner("Finding related questions in your files..."):
                            reply["suggestions"] = suggest_questions(
                                history + [{"role": "user", "content": question}], hits)
                        show_suggestions(reply["suggestions"], key=f"s{len(history) + 1}")

            reply["content"] = answer
        except Exception as exc:  # show a friendly message instead of crashing the app
            reply["content"] = f"Sorry, something went wrong: {exc}"
            st.error(reply["content"])

    st.session_state.messages += [{"role": "user", "content": question}, reply]
