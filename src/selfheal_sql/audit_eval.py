"""AuditKIT agent_eval over a student run: independent outcome verification.

Each logged run row becomes a recorded AgentEpisode (user prompt -> assistant raw
output). The outcome oracle is a CustomPredicate that RE-EXECUTES the model's SQL
against the real database and compares with the gold result - AuditKIT's verdict is a
fresh execution, not a read of our logged status. We then cross-check AuditKIT's
verdicts against the harness log (they must agree 100%).

Usage: python audit_eval.py --run baseline-test --split-file ../../data/splits/test.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from execcheck import check  # noqa: E402
from student import PROMPT  # noqa: E402

RUNS_REPO = "HideIron/selfheal-sql-runs"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--split-file", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-push", action="store_true")
    args = ap.parse_args()

    from huggingface_hub import HfApi, hf_hub_download

    from auditkit.agent_eval import (
        AgentCase,
        AgentEvalRunner,
        AgentEvalSpec,
        CustomPredicate,
        episode_from_openai_messages,
    )

    rp = hf_hub_download(RUNS_REPO, f"runs/{args.run}/results.jsonl", repo_type="dataset")
    rows = [json.loads(l) for l in open(rp, encoding="utf-8")]
    splits = {r["qid"]: r for r in map(json.loads, open(args.split_file, encoding="utf-8"))}

    def exec_predicate(episode, case):
        s = splits[case.id]
        res = check(s["db_path"], episode.final_answer or "", s["gold_sql"])
        return bool(res.correct)

    episodes, cases = [], []
    for r in rows:
        s = splits[r["qid"]]
        episodes.append(
            episode_from_openai_messages(
                [
                    {"role": "user",
                     "content": PROMPT.format(schema=s["schema"], question=s["question"])},
                    {"role": "assistant", "content": r["raw_output"]},
                ],
                case_id=r["qid"],
                final_answer=r["raw_output"],
            )
        )
        cases.append(
            AgentCase(
                id=r["qid"], task=s["question"], category="text_to_sql",
                outcome=CustomPredicate(exec_predicate, version="execcheck-v1",
                                        name="spider_execution_accuracy"),
                metadata={"db_id": s["db_id"]},
            )
        )

    spec = AgentEvalSpec(cases=cases, mode="recorded", episodes=episodes,
                         scorers=[], trials=1)
    result = AgentEvalRunner().run(spec)

    verdicts = {row.case_id: row.outcome["verdict"] for row in result.rows}
    logged = {r["qid"]: ("success" if r["status"] == "correct" else "failure") for r in rows}
    agree = sum(verdicts[q] == logged[q] for q in verdicts)
    out = {
        "run": args.run,
        "auditkit_summary": result.summary(),
        "verdict_counts": dict(Counter(verdicts.values())),
        "n": len(verdicts),
        "success_rate": round(sum(v == "success" for v in verdicts.values()) / len(verdicts), 4),
        "crosscheck_agree_with_harness": f"{agree}/{len(verdicts)}",
        "oracle": "CustomPredicate(spider_execution_accuracy, version=execcheck-v1) - fresh re-execution",
    }
    out_dir = args.out or os.path.join(HERE, "..", "..", "data", "audit", args.run)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "auditkit_report.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    print(json.dumps(out, indent=2, default=str))

    if not args.no_push:
        HfApi().upload_folder(repo_id=RUNS_REPO, repo_type="dataset", folder_path=out_dir,
                              path_in_repo=f"audit/{args.run}",
                              commit_message=f"auditkit report {args.run}")


if __name__ == "__main__":
    main()
