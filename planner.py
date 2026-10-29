"""
Query planner: understand the question FIRST, then let code write the SQL.

Why? Small local models make subtle SQL mistakes, e.g. "how many salespeople?" written as
COUNT(salesperson) = number of ORDER ROWS (563) instead of COUNT(DISTINCT salesperson) = 11.
Filling in a small form (a "plan") is much easier for a small model than writing SQL, and
code can check every field of the form against the real columns before anything runs.

    question --LLM--> plan (JSON form) --validate--> build_sql() --> correct, safe SQL
                       │ invalid / too complex
                       └──────────────▶ fall back to free-form SQL (data_agent)
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

METRICS = ("count_rows", "count_distinct", "sum", "avg", "min", "max", "list")
OPS = ("=", "!=", ">", ">=", "<", "<=", "contains")
NUMERIC = ("INT", "DOUBLE", "FLOAT", "DECIMAL", "REAL", "NUMERIC")

PLAN_PROMPT = """Turn the user's question into a query PLAN. First read the column notes:
they show what each column contains and its exact values.

{schema}

Reply with JSON only, in exactly this shape:
{{"table": "<table>", "metric": "<metric>", "column": "<column or math, or null>",
  "group_by": "<column or null>", "filters": [{{"column": "<column>", "op": "<op>", "value": "<value>"}}],
  "date_column": "<column or null>", "date_from": "YYYY-MM-DD or null", "date_to": "YYYY-MM-DD or null",
  "order": "desc | asc | null", "limit": <number or null>, "complex": false}}

metric is one of:
- count_rows     -> how many ROWS/records/orders/transactions ("how many orders")
- count_distinct -> how many DIFFERENT things ("how many salespeople / customers / products / regions")
- sum, avg, min, max -> of a number column, or of simple math on number columns ("units * unit_price")
- list           -> show the different values of a column, or show matching rows

Rules:
- "how many <people or things>" is count_distinct of that column. Only orders/rows/records use count_rows.
- "most / top / highest / best" -> order "desc" with a limit (1 unless a number is given).
  "least / lowest / worst" -> order "asc". "per X" / "by X" / "each X" -> group_by X.
- filters: op is one of = != > >= < <= contains. Use text values exactly as in the notes.
- Use only columns from ONE table. If the question really needs two tables, a subquery,
  or several different metrics, set "complex": true.

Examples (for a different table "orders" with columns customer, city, qty, price, order_day):
"How many customers do we have?" -> {{"table": "orders", "metric": "count_distinct", "column": "customer", "group_by": null, "filters": [], "date_column": null, "date_from": null, "date_to": null, "order": null, "limit": null, "complex": false}}
"Which city sold the most quantity in March 2025?" -> {{"table": "orders", "metric": "sum", "column": "qty", "group_by": "city", "filters": [], "date_column": "order_day", "date_from": "2025-03-01", "date_to": "2025-03-31", "order": "desc", "limit": 1, "complex": false}}
"Revenue per customer in Pune" -> {{"table": "orders", "metric": "sum", "column": "qty * price", "group_by": "customer", "filters": [{{"column": "city", "op": "=", "value": "Pune"}}], "date_column": null, "date_from": null, "date_to": null, "order": "desc", "limit": null, "complex": false}}"""


class PlanError(ValueError):
    """The plan doesn't match the real tables/columns, or needs free-form SQL."""


@dataclass
class Plan:
    table: str
    metric: str
    column: str | None = None
    group_by: str | None = None
    filters: list[dict] = field(default_factory=list)
    date_column: str | None = None
    date_from: str | None = None
    date_to: str | None = None
    order: str | None = None
    limit: int | None = None

    def describe(self) -> str:
        """Plain-English summary shown to the user ('Understood as ...')."""
        what = {"count_rows": "count of rows", "count_distinct": f"number of different {self.column}",
                "list": f"list of {self.column or 'rows'}"}.get(self.metric, f"{self.metric} of {self.column}")
        parts = [what]
        if self.group_by:
            parts.append(f"per {self.group_by}")
        parts += [f"where {f['column']} {f['op']} {f['value']}" for f in self.filters]
        if self.date_column and (self.date_from or self.date_to):
            parts.append(f"{self.date_column} from {self.date_from or 'start'} to {self.date_to or 'end'}")
        if self.order and self.limit:
            parts.append(f"{'top' if self.order == 'desc' else 'bottom'} {self.limit}")
        return ", ".join(parts)


def _none(v):
    return None if v in (None, "", "null", "None") else v


def parse_plan(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise PlanError("No JSON plan in the reply")
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError as exc:
        raise PlanError(f"Bad JSON: {exc}") from exc


def validate_plan(raw: dict, tables: list) -> tuple[Plan, object]:
    """Check every field against the real table. Returns (plan, table) or raises PlanError."""
    if raw.get("complex") is True:
        raise PlanError("Question needs free-form SQL")
    by_name = {t.name: t for t in tables}
    table = by_name.get(str(raw.get("table", "")).strip('"')) or (tables[0] if len(tables) == 1 else None)
    if table is None:
        raise PlanError(f"Unknown table: {raw.get('table')}")
    types = dict(zip(table.columns, table.types))

    def col(name, required=False):
        name = _none(name)
        if name is None:
            if required:
                raise PlanError("Missing column")
            return None
        name = str(name).strip('"')
        if name not in types:
            raise PlanError(f"Unknown column: {name}")
        return name

    metric = str(raw.get("metric", "")).lower()
    if metric not in METRICS:
        raise PlanError(f"Unknown metric: {metric}")
    column = _none(raw.get("column"))
    if metric in ("sum", "avg", "min", "max"):
        column = _math(column, types)            # "units * unit_price" etc., numbers only
    elif metric == "count_distinct":
        column = col(column, required=True)
    elif metric == "list":
        column = col(column)
    else:
        column = None

    filters = []
    for f in raw.get("filters") or []:
        op = str(f.get("op", "=")).strip()
        if op not in OPS:
            raise PlanError(f"Unknown operator: {op}")
        filters.append({"column": col(f.get("column"), required=True), "op": op, "value": f.get("value")})

    date_column = col(raw.get("date_column"))
    dates = {}
    for k in ("date_from", "date_to"):
        v = _none(raw.get(k))
        if v is not None and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(v)):
            raise PlanError(f"Bad date: {v}")
        dates[k] = v
    order = _none(raw.get("order"))
    if order not in (None, "asc", "desc"):
        raise PlanError(f"Bad order: {order}")
    limit = _none(raw.get("limit"))
    limit = int(limit) if limit is not None else None
    if limit is not None and not 1 <= limit <= 1000:
        raise PlanError(f"Bad limit: {limit}")
    plan = Plan(table=table.name, metric=metric, column=column, group_by=col(raw.get("group_by")),
                filters=filters, date_column=date_column, order=order, limit=limit, **dates)
    return plan, table


def _math(expr, types: dict) -> str:
    """Allow a number column, or simple math on number columns: units * unit_price * (1 - discount_pct / 100)."""
    expr = _none(expr)
    if expr is None:
        raise PlanError("This metric needs a number column")
    expr = str(expr).replace('"', "")
    if not re.fullmatch(r"[\w\s.*/+\-()]+", expr):
        raise PlanError(f"Unsupported math: {expr}")
    for name in re.findall(r"[A-Za-z_]\w*", expr):
        if name not in types:
            raise PlanError(f"Unknown column: {name}")
        if not any(k in types[name] for k in NUMERIC):
            raise PlanError(f"Not a number column: {name}")
    return expr


def _ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _value(v) -> str:
    """Numbers stay numbers; everything else becomes a safely quoted SQL string."""
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return str(v)
    s = str(v)
    if re.fullmatch(r"-?\d+(\.\d+)?", s):
        return s
    return "'" + s.replace("'", "''") + "'"


def build_sql(plan: Plan, table) -> str:
    """Plan -> SQL. Written by code, so syntax, quoting and DISTINCT are always right."""
    types = dict(zip(table.columns, table.types))
    math = re.sub(r"[A-Za-z_]\w*", lambda m: _ident(m.group(0)), plan.column) if plan.metric in (
        "sum", "avg", "min", "max") else None
    alias = re.sub(r"[^a-z0-9]+", "_", f"{plan.metric}_{plan.column or 'rows'}".lower()).strip("_")[:40]
    metric_sql = {
        "count_rows": "COUNT(*) AS row_count",
        "count_distinct": f"COUNT(DISTINCT {_ident(plan.column or '')}) AS {alias}",
        "sum": f"SUM({math}) AS {alias}", "avg": f"ROUND(AVG({math}), 2) AS {alias}",
        "min": f"MIN({math}) AS {alias}", "max": f"MAX({math}) AS {alias}",
    }
    where = []
    for f in plan.filters:
        c, op, v = _ident(f["column"]), f["op"], f["value"]
        is_text = types.get(f["column"], "").startswith("VARCHAR")
        if op == "contains":
            where.append(f"{c} ILIKE {_value('%' + str(v) + '%')}")
        elif is_text and op in ("=", "!="):
            where.append(f"{c} {'NOT ' if op == '!=' else ''}ILIKE {_value(v)}")   # case-insensitive
        else:
            where.append(f"{c} {op} {_value(v)}")
    if plan.date_column:
        d = _ident(plan.date_column)
        if plan.date_from:
            where.append(f"{d} >= '{plan.date_from}'")
        if plan.date_to:
            where.append(f"{d} < '{plan.date_to}'::DATE + INTERVAL 1 DAY")   # include the whole last day
    where_sql = f" WHERE {' AND '.join(where)}" if where else ""
    tbl = _ident(table.name)

    if plan.metric == "list":
        cols = [c for c in (plan.group_by, plan.column) if c] or ["*"]
        select = ", ".join(_ident(c) if c != "*" else c for c in dict.fromkeys(cols))
        distinct = "DISTINCT " if cols != ["*"] else ""
        order = f" ORDER BY 1 {plan.order.upper()}" if plan.order else (" ORDER BY 1" if distinct else "")
        limit = f" LIMIT {plan.limit}" if plan.limit else ""
        return f"SELECT {distinct}{select} FROM {tbl}{where_sql}{order}{limit}"

    if plan.group_by:
        g = _ident(plan.group_by)
        order = f" ORDER BY 2 {plan.order.upper()}" if plan.order else " ORDER BY 1"
        limit = f" LIMIT {plan.limit}" if plan.limit else ""
        return f"SELECT {g}, {metric_sql[plan.metric]} FROM {tbl}{where_sql} GROUP BY 1{order}{limit}"
    return f"SELECT {metric_sql[plan.metric]} FROM {tbl}{where_sql}"
