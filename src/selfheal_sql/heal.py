"""The heal step: classify each failure, generate a corrected SQL query, keep it only
if it ACTUALLY produces the gold result. Output: DPO preference pairs.

Replaces AgentTune's TrainingExampleGenerator, which is tool-call-specific (it drops
every failure whose root cause isn't wrong_tool/loop_collapse, asks the LLM for "the
EXACT JSON tool call", and trains on a placeholder prompt). The classification stage
keeps AgentTune's FailureClassifier contract ({root_cause, confidence, analysis}) but
with a SQL-specific taxonomy. The output rows carry chosen/rejected exactly as
agenttune.decide.closed_loop.contracts.TrainingExample expects, so the retrain step
consumes them through Lexsi's own examples_to_dpo_dataset / run_retrain.

The healer NEVER sees the gold SQL. Gold is used only to verify a proposed fix by
executing both and comparing results (execcheck). Failed fixes are retried with the
new error message, up to --max-attempts.

Runs locally (needs the Spider DBs on disk + the HF router for LLM calls).

Usage:
  python heal.py --run baseline-pool600 --split-file ../../data/splits/pool.jsonl \
      --out ../../data/heal [--limit 6] [--workers 6]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from execcheck import check, extract_sql  # noqa: E402
from student import PROMPT  # noqa: E402  (the student's prompt - reused verbatim)

ROUTER = "https://router.huggingface.co/v1/chat/completions"
HEALER_MODEL = "CohereLabs/aya-expanse-32b"  # on the hackathon's allowed model list
RUNS_REPO = "HideIron/selfheal-sql-runs"
SCHEMA_CAP = 7000  # chars of schema shown to the healer

# SQL-specific root causes (replaces the tool-agent taxonomy in
# agenttune FailureClassifier.ROOT_CAUSES; same output contract).
ROOT_CAUSES = [
    "schema_misread",       # non-existent column/table, column on wrong table
    "wrong_join",           # wrong join path, missing join, wrong join keys
    "wrong_aggregation",    # count/sum/avg/group-by/having misuse
    "missing_or_wrong_filter",  # WHERE/HAVING condition missing or wrong
    "value_mismatch",       # wrong literal (e.g. 'US' vs 'United States')
    "wrong_output_columns", # right rows, wrong selected columns/order/distinct
    "overcomplicated_query",  # needless subquery/join that changes the result
    "other",
]

_counts_lock = threading.Lock()
TOKENS = [0, 0]  # prompt, completion
CALLS = [0]


def llm(messages: list[dict], temperature: float, max_tokens: int = 500) -> str:
    """One chat call to the HF router, with backoff on 429/5xx. Returns content."""
    body = json.dumps(
        {"model": HEALER_MODEL, "messages": messages, "temperature": temperature,
         "max_tokens": max_tokens}
    ).encode()
    last = None
    for i in range(5):
        req = urllib.request.Request(
            ROUTER, data=body,
            headers={"Authorization": "Bearer " + os.environ["HF_TOKEN"],
                     "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                d = json.load(r)
            u = d.get("usage", {})
            with _counts_lock:
                TOKENS[0] += u.get("prompt_tokens", 0)
                TOKENS[1] += u.get("completion_tokens", 0)
                CALLS[0] += 1
            return d["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code}: {e.read()[:200]!r}"
            if e.code in (429, 500, 502, 503, 504):
                time.sleep(2 ** i)
                continue
            raise RuntimeError(last) from e
        except Exception as e:  # noqa: BLE001  (timeouts, resets)
            last = repr(e)
            time.sleep(2 ** i)
    raise RuntimeError(f"router call failed after retries: {last}")


def _json_block(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    return json.loads(m.group(0)) if m else {}


def classify(fail: dict, srow: dict) -> dict:
    """SQL-specific root-cause classification (AgentTune ClassifiedFailure contract)."""
    sys_p = (
        "You are an expert SQL failure analyst. Classify the failure into EXACTLY ONE "
        "root cause from this list:\n" + ", ".join(ROOT_CAUSES) + "\n\n"
        'Return ONLY a JSON object: {"root_cause": "<one of the list>", '
        '"confidence": 0.0-1.0, "analysis": "one short sentence"}'
    )
    user = (
        f"Schema:\n{srow['schema'][:SCHEMA_CAP]}\n\nQuestion: {srow['question']}\n\n"
        f"Model's SQL:\n{fail['raw_output'][:1200]}\n\n"
        f"Outcome: {fail['status']}"
        + (f" - {fail['error_message']}" if fail.get("error_message") else "")
        + (
            f" (query returned {fail['pred_rows']} rows, expected result has {fail['gold_rows']} rows)"
            if fail.get("pred_rows") is not None and fail.get("gold_rows") is not None
            else ""
        )
    )
    try:
        p = _json_block(llm([{"role": "system", "content": sys_p},
                             {"role": "user", "content": user}], 0.0, 250))
        rc = p.get("root_cause", "other")
        return {
            "root_cause": rc if rc in ROOT_CAUSES else "other",
            "confidence": float(p.get("confidence", 0.5)),
            "analysis": str(p.get("analysis", ""))[:400],
            "classify_error": None,
        }
    except Exception as e:  # noqa: BLE001
        return {"root_cause": "other", "confidence": 0.0, "analysis": "",
                "classify_error": str(e)[:300]}


def generate_fix(fail: dict, srow: dict, cls: dict, max_attempts: int) -> tuple[str | None, list]:
    """Ask the healer for corrected SQL; keep only a fix that executes to the gold
    result. Retries feed the candidate's own error back. Returns (sql|None, attempts)."""
    base = (
        f"A SQLite query is wrong and you must fix it.\n\n"
        f"Schema:\n{srow['schema'][:SCHEMA_CAP]}\n\n"
        f"Question: {srow['question']}\n\n"
        f"Wrong SQL:\n{fail['raw_output'][:1200]}\n\n"
        f"Observed problem: {fail['status']}"
        + (f" - {fail['error_message']}" if fail.get("error_message") else "")
        + (f"\nLikely root cause: {cls['root_cause']} ({cls['analysis']})" if cls.get("analysis") else "")
        + "\n\nRules: make the MINIMAL edit that fixes the query. Keep the original's "
        "formatting, aliasing style, capitalisation and column order; do not add joins, "
        "aliases, extra columns or reformat. Change only what is wrong.\n"
        "Return ONLY the corrected SQLite query, nothing else."
    )
    msgs = [{"role": "user", "content": base}]
    attempts = []
    for k in range(max_attempts):
        try:
            out = llm(msgs, 0.0 if k == 0 else 0.4, 400)
        except Exception as e:  # noqa: BLE001
            attempts.append({"attempt": k + 1, "error": f"llm: {str(e)[:200]}"})
            break
        cand = extract_sql(out)
        if not cand:
            attempts.append({"attempt": k + 1, "error": "no SQL in healer output"})
            feedback = "Your reply contained no SQL. Return ONLY the corrected SQLite query."
        else:
            res = check(srow["db_path"], cand, srow["gold_sql"])
            attempts.append({"attempt": k + 1, "sql": cand, "status": res.status,
                             "error": res.error_message[:200]})
            if res.correct:
                return cand, attempts
            feedback = (
                f"Your query:\n{cand}\n\nIt is still wrong: {res.status}"
                + (f" - {res.error_message[:300]}" if res.error_message else "")
                + (
                    f" (it returned {res.pred_rows} rows; the correct result has {res.gold_rows} rows)"
                    if res.pred_rows is not None and res.gold_rows is not None
                    else ""
                )
                + "\nTry again. Return ONLY the corrected SQLite query."
            )
        msgs = msgs + [{"role": "assistant", "content": out},
                       {"role": "user", "content": feedback}]
    return None, attempts


def heal_one(fail: dict, srow: dict, max_attempts: int) -> dict:
    cls = classify(fail, srow)
    fixed_sql, attempts = generate_fix(fail, srow, cls, max_attempts)
    rejected = extract_sql(fail["raw_output"]) or fail["raw_output"].strip()
    return {
        "qid": fail["qid"], "db_id": fail["db_id"], "question": srow["question"],
        "status_before": fail["status"], "error_before": fail.get("error_message", ""),
        **{k: cls[k] for k in ("root_cause", "confidence", "analysis", "classify_error")},
        "healed": fixed_sql is not None,
        "attempts_used": len(attempts), "attempts": attempts,
        # --- TrainingExample-ready fields (consumed by train_dpo.py) ---
        "prompt_user_content": PROMPT.format(schema=srow["schema"], question=srow["question"]),
        "chosen_sql": fixed_sql,
        "rejected_sql": rejected,
        "healer_model": HEALER_MODEL,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="baseline-pool600", help="runs-repo folder with results.jsonl")
    ap.add_argument("--split-file", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-attempts", type=int, default=3)
    ap.add_argument("--no-push", action="store_true")
    ap.add_argument("--tag", default="", help="suffix for the runs-repo push path")
    args = ap.parse_args()

    from huggingface_hub import HfApi, hf_hub_download

    rp = hf_hub_download(RUNS_REPO, f"runs/{args.run}/results.jsonl", repo_type="dataset")
    results = [json.loads(l) for l in open(rp, encoding="utf-8")]
    splits = {r["qid"]: r for r in map(json.loads, open(args.split_file, encoding="utf-8"))}
    fails = [r for r in results if r["status"] != "correct"]
    if args.limit:
        fails = fails[: args.limit]
    print(f"[heal] {len(fails)} failures to heal (of {len(results)} runs)", flush=True)

    t0 = time.time()
    done = [0]

    def work(f):
        row = heal_one(f, splits[f["qid"]], args.max_attempts)
        with _counts_lock:
            done[0] += 1
            if done[0] % 20 == 0:
                print(f"[heal] {done[0]}/{len(fails)} ({time.time()-t0:.0f}s, "
                      f"{CALLS[0]} calls)", flush=True)
        return row

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        rows = list(ex.map(work, fails))

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "pairs.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    healed = [r for r in rows if r["healed"]]
    summary = {
        "run": args.run, "healer_model": HEALER_MODEL, "n_failures": len(fails),
        "n_healed": len(healed), "heal_yield": round(len(healed) / max(len(fails), 1), 4),
        "by_root_cause": dict(Counter(r["root_cause"] for r in rows)),
        "healed_by_root_cause": dict(Counter(r["root_cause"] for r in healed)),
        "attempts_histogram": dict(Counter(r["attempts_used"] for r in healed)),
        "classify_errors": sum(1 for r in rows if r["classify_error"]),
        "llm_calls": CALLS[0], "prompt_tokens": TOKENS[0], "completion_tokens": TOKENS[1],
        "elapsed_s": round(time.time() - t0, 1), "max_attempts": args.max_attempts,
    }
    with open(os.path.join(args.out, "heal_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("[heal] SUMMARY", json.dumps(summary), flush=True)

    if not args.no_push:
        dest = f"heal/{args.run}{args.tag}"
        HfApi().upload_folder(repo_id=RUNS_REPO, repo_type="dataset", folder_path=args.out,
                              path_in_repo=dest, commit_message=f"heal {dest}")
        print("[heal] pushed to", RUNS_REPO, dest, flush=True)


if __name__ == "__main__":
    main()
