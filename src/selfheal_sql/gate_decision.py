"""Deployment gate: decide whether the retrained adapter ships, using AgentTune's
DeploymentGate verbatim (vendored at third_party/agenttune_gate, commit 1dfd054).

The gate is model-agnostic by design: `model_runner_fn(input_text) -> verdict` is
supplied by the caller and compared against `expected_verdict` per case ("keeps the
gate model-agnostic and testable without a GPU" - their docstring). Our runners look
up each gate question's execution-checker outcome from the student harness's logged
runs, so the verdict IS Spider execution accuracy.

Usage:
  python gate_decision.py --old baseline-gate --new healed-gate --out ../../data/gate
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import types
from dataclasses import asdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
RUNS_REPO = "HideIron/selfheal-sql-runs"


def load_gate_module():
    """Import the vendored deployment_gate.py without installing agenttune
    (its only non-stdlib import is the PASS_VERDICTS constant, stubbed here 1:1)."""
    pkg_names = ["agenttune", "agenttune.decide"]
    for n in pkg_names:
        if n not in sys.modules:
            m = types.ModuleType(n)
            m.__path__ = []  # mark as package
            sys.modules[n] = m
    st = types.ModuleType("agenttune.decide.state")
    st.PASS_VERDICTS = frozenset({"PASS", "APPROVE", "COMPLETE"})  # verbatim
    sys.modules["agenttune.decide.state"] = st
    path = os.path.join(ROOT, "third_party", "agenttune_gate", "deployment_gate.py")
    spec = importlib.util.spec_from_file_location("deployment_gate", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["deployment_gate"] = mod  # dataclasses resolves cls.__module__ here
    spec.loader.exec_module(mod)
    return mod


def results_lookup(run_name: str) -> dict[str, str]:
    from huggingface_hub import hf_hub_download

    p = hf_hub_download(RUNS_REPO, f"runs/{run_name}/results.jsonl", repo_type="dataset")
    out = {}
    for line in open(p, encoding="utf-8"):
        r = json.loads(line)
        out[r["qid"]] = "CORRECT" if r["status"] == "correct" else r["status"].upper()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", default="baseline-gate")
    ap.add_argument("--new", default="healed-gate")
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "gate"))
    ap.add_argument("--task-regression-tol", type=float, default=0.0)
    ap.add_argument("--no-push", action="store_true")
    args = ap.parse_args()

    gate_mod = load_gate_module()
    old_v, new_v = results_lookup(args.old), results_lookup(args.new)
    qids = sorted(old_v)
    assert set(qids) == set(new_v), "old/new runs cover different questions"

    test_set = [{"pipeline_id": q, "input_text": q, "expected_verdict": "CORRECT"}
                for q in qids]

    gate = gate_mod.DeploymentGate()
    decision = gate.evaluate_decision(
        test_set=test_set,
        old_model_fn=lambda qid: old_v[qid],
        new_model_fn=lambda qid: new_v[qid],
        task_regression_tol=args.task_regression_tol,
    )
    comparison = gate.compare_models(
        test_set=test_set,
        old_model_fn=lambda qid: old_v[qid],
        new_model_fn=lambda qid: new_v[qid],
    )

    out = {
        "gate_decision": asdict(decision),
        "ab_comparison": asdict(comparison),
        "old_run": args.old, "new_run": args.new,
        "per_question_flips": {
            "fixed": [q for q in qids if old_v[q] != "CORRECT" and new_v[q] == "CORRECT"],
            "broken": [q for q in qids if old_v[q] == "CORRECT" and new_v[q] != "CORRECT"],
        },
        "gate_source": "AgentTune deployment_gate.py @ 1dfd054 (vendored verbatim)",
    }
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "gate_decision.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps({k: out[k] for k in ("gate_decision", "ab_comparison")}, indent=2))
    print("fixed:", len(out["per_question_flips"]["fixed"]),
          "broken:", len(out["per_question_flips"]["broken"]))

    if not args.no_push:
        from huggingface_hub import HfApi

        HfApi().upload_folder(repo_id=RUNS_REPO, repo_type="dataset", folder_path=args.out,
                              path_in_repo="gate", commit_message="gate decision")


if __name__ == "__main__":
    main()
