"""Final before/after metrics for the write-up. Reads student-run results from the
runs repo, computes per-split accuracy, status breakdowns, heal/regression rates and
a paired significance test (McNemar exact), and writes one metrics.json.

Usage: python metrics.py --pairs before:after [...] e.g.
  python metrics.py --pair test:baseline-test:healed-test --pair gate:baseline-gate:healed-gate
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter

RUNS_REPO = "HideIron/selfheal-sql-runs"


def load(run: str) -> dict[str, dict]:
    from huggingface_hub import hf_hub_download

    p = hf_hub_download(RUNS_REPO, f"runs/{run}/results.jsonl", repo_type="dataset")
    return {r["qid"]: r for r in map(json.loads, open(p, encoding="utf-8"))}


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value. b = fixed (wrong->right), c = broken."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / 2**n
    return min(1.0, 2 * tail)


def split_metrics(before: dict, after: dict) -> dict:
    qids = sorted(before)
    assert set(qids) == set(after)
    n = len(qids)
    acc_b = sum(before[q]["status"] == "correct" for q in qids) / n
    acc_a = sum(after[q]["status"] == "correct" for q in qids) / n
    fixed = [q for q in qids if before[q]["status"] != "correct" and after[q]["status"] == "correct"]
    broken = [q for q in qids if before[q]["status"] == "correct" and after[q]["status"] != "correct"]
    n_fail_before = sum(before[q]["status"] != "correct" for q in qids)
    n_corr_before = n - n_fail_before

    def valid_rate(d):
        return sum(d[q]["status"] in ("correct", "wrong_result") for q in qids) / n

    return {
        "n": n,
        "execution_accuracy_before": round(acc_b, 4),
        "execution_accuracy_after": round(acc_a, 4),
        "delta_pp": round((acc_a - acc_b) * 100, 2),
        "valid_sql_rate_before": round(valid_rate(before), 4),
        "valid_sql_rate_after": round(valid_rate(after), 4),
        "status_before": dict(Counter(before[q]["status"] for q in qids)),
        "status_after": dict(Counter(after[q]["status"] for q in qids)),
        "n_fixed": len(fixed),
        "n_broken": len(broken),
        "heal_rate": round(len(fixed) / n_fail_before, 4) if n_fail_before else None,
        "regression_rate": round(len(broken) / n_corr_before, 4) if n_corr_before else None,
        "mcnemar_p": round(mcnemar_exact(len(fixed), len(broken)), 6),
        "fixed_qids": fixed,
        "broken_qids": broken,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", action="append", required=True,
                    help="label:before_run:after_run (repeatable)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-push", action="store_true")
    args = ap.parse_args()

    report = {}
    for spec in args.pair:
        label, b, a = spec.split(":")
        report[label] = {"before_run": b, "after_run": a, **split_metrics(load(b), load(a))}
        r = report[label]
        print(f"[{label}] {r['execution_accuracy_before']:.1%} -> "
              f"{r['execution_accuracy_after']:.1%} ({r['delta_pp']:+.1f}pp, "
              f"fixed {r['n_fixed']}, broke {r['n_broken']}, p={r['mcnemar_p']})")

    out_dir = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       "..", "..", "data", "metrics")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "metrics.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print("wrote", path)
    if not args.no_push:
        from huggingface_hub import HfApi

        HfApi().upload_folder(repo_id=RUNS_REPO, repo_type="dataset", folder_path=out_dir,
                              path_in_repo="metrics", commit_message="final metrics")


if __name__ == "__main__":
    main()
