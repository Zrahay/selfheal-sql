"""Execution checker for text-to-SQL: the single source of truth for pass/fail.

A predicted query is CORRECT iff executing it on the question's SQLite database
returns the same result as the gold query (Spider-style execution accuracy).

Outcome statuses (feed the self-heal Failure objects):
  correct       - result matches gold
  wrong_result  - ran fine, result differs from gold
  sql_error     - sqlite raised (bad column/table/syntax ...); error_message holds it
  timeout       - exceeded the time budget (likely a runaway / cartesian join)
  empty_sql     - nothing usable was extracted from the model output
"""

from __future__ import annotations

import re
import sqlite3
import time
from dataclasses import asdict, dataclass

TIMEOUT_S = 10.0
MAX_ROWS = 100_000  # guard against pathological result sets


@dataclass
class ExecResult:
    status: str  # correct | wrong_result | sql_error | timeout | empty_sql
    error_message: str = ""
    pred_rows: int | None = None
    gold_rows: int | None = None

    @property
    def correct(self) -> bool:
        return self.status == "correct"

    def to_dict(self) -> dict:
        return asdict(self)


_FENCE = re.compile(r"```(?:sql)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def extract_sql(text: str) -> str:
    """Pull a SQL statement out of raw model output.

    Handles <think> blocks, ```sql fences, and a trailing explanation. Returns ""
    if nothing that looks like SQL is found.
    """
    if not text:
        return ""
    text = _THINK.sub("", text)
    m = _FENCE.search(text)
    if m:
        text = m.group(1)
    text = text.strip()
    # Start at the first SELECT/WITH; drop anything before it.
    m = re.search(r"\b(select|with)\b", text, re.IGNORECASE)
    if not m:
        return ""
    text = text[m.start():]
    # Keep only the first statement.
    text = text.split(";")[0].strip()
    return text


def _connect(db_path: str) -> sqlite3.Connection:
    # Read-only: a model-written query must never modify the benchmark databases.
    uri = "file:" + db_path.replace("\\", "/") + "?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.text_factory = lambda b: b.decode("utf-8", errors="ignore")
    return conn


def _run(conn: sqlite3.Connection, sql: str, timeout_s: float):
    deadline = time.monotonic() + timeout_s

    def _abort() -> int:
        return 1 if time.monotonic() > deadline else 0

    conn.set_progress_handler(_abort, 10_000)
    try:
        cur = conn.execute(sql)
        rows = cur.fetchmany(MAX_ROWS)
    finally:
        conn.set_progress_handler(None, 0)
    return rows


def _norm_cell(v):
    if isinstance(v, float):
        return round(v, 6)
    return v


def _norm_rows(rows) -> list[tuple]:
    return [tuple(_norm_cell(c) for c in r) for r in rows]


def _results_match(pred, gold, ordered: bool) -> bool:
    pred, gold = _norm_rows(pred), _norm_rows(gold)
    if ordered:
        return pred == gold
    key = lambda r: tuple(repr(c) for c in r)  # noqa: E731  (mixed types sort safely)
    return sorted(pred, key=key) == sorted(gold, key=key)


def check(db_path: str, pred_sql: str, gold_sql: str, timeout_s: float = TIMEOUT_S) -> ExecResult:
    """Execute pred and gold on db_path and compare. Never raises."""
    # Always extract: strips fences/<think>/trailing prose and keeps only the FIRST
    # statement (a model that appends "; <explanation>" must not be marked wrong).
    pred_sql = extract_sql(pred_sql)
    if not pred_sql:
        return ExecResult("empty_sql", "no SQL found in model output")

    try:
        conn = _connect(db_path)
    except Exception as e:  # noqa: BLE001
        return ExecResult("sql_error", f"cannot open database: {e}")

    try:
        try:
            gold = _run(conn, gold_sql, timeout_s * 3)
        except Exception as e:  # noqa: BLE001
            return ExecResult("sql_error", f"GOLD query failed (bad benchmark row): {e}")
        try:
            pred = _run(conn, pred_sql, timeout_s)
        except sqlite3.OperationalError as e:
            msg = str(e)
            if "interrupted" in msg.lower():
                return ExecResult("timeout", f"query exceeded {timeout_s}s", None, len(gold))
            return ExecResult("sql_error", msg, None, len(gold))
        except Exception as e:  # noqa: BLE001
            return ExecResult("sql_error", f"{type(e).__name__}: {e}", None, len(gold))

        ordered = bool(re.search(r"\border\s+by\b", gold_sql, re.IGNORECASE))
        ok = _results_match(pred, gold, ordered)
        return ExecResult("correct" if ok else "wrong_result", "", len(pred), len(gold))
    finally:
        conn.close()


def schema_text(db_path: str) -> str:
    """CREATE TABLE statements for the prompt (the schema the agent is shown)."""
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL ORDER BY name"
        ).fetchall()
    finally:
        conn.close()
    return "\n".join(r[0].strip() + ";" for r in rows)
