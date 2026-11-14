# Self-Healing Text-to-SQL (Lexsi Hackathon, Track 3, PS #8)

A fully automated self-healing loop for a **Tiny Aya Global (3.35B)** text-to-SQL
agent on **Spider**: detect failing runs by *executing* the SQL, classify root causes,
generate **execution-verified** corrective training pairs with **Aya Expanse 32B**,
retrain adapter-only (DPO / BCO) through **AgentTune**, and gate redeployment with
**AgentTune's DeploymentGate**. Outcomes independently verified with **AuditKIT**.

Result: the gate blocked four regressing adapters, approved the fifth
(BCO + success replay) on a 400-question held-out gate, and the deployed adapter
improves the unseen 500-question test split 45.6% -> 47.6% execution accuracy
(valid-SQL 68.2% -> 73.0%).

```
student (Tiny Aya) writes SQL ──▶ execcheck runs it vs gold ──▶ failures
     ▲                                                             │
     │                                          classify (SQL taxonomy, Aya-32B)
 LoRA adapter                                                      │
 (AgentTune DPO/BCO retrain)                    generate fix (Aya-32B, never sees gold)
     ▲                                                             │
     │                                          execute fix ─ correct? keep : retry ≤3
 158 verified pairs ◀──────────────────────────────────────────────┘
     │
 DeploymentGate (AgentTune, held-out gate split) ──▶ deploy / keep old
     │
 final report on unseen test split (+ AuditKIT re-verification)
```

## Layout

| path | what |
|---|---|
| `src/selfheal_sql/execcheck.py` | Execution checker, the single source of truth for pass/fail |
| `src/selfheal_sql/make_splits.py` | Database-disjoint pool/gate/test splits from Spider |
| `src/selfheal_sql/make_gate_large.py` | Enlarges the gate split 100 -> 400 for statistical power |
| `src/selfheal_sql/student.py` | Student harness: prompt -> SQL -> execute -> grade -> log |
| `src/selfheal_sql/heal.py` | Classify failures + generate execution-verified training pairs |
| `src/selfheal_sql/train_dpo.py` | Adapter-only DPO/BCO via AgentTune `RetrainConfig`/`build_retrainer` |
| `src/selfheal_sql/gate_decision.py` | AgentTune `DeploymentGate.evaluate_decision` on gate data |
| `src/selfheal_sql/audit_eval.py` | AuditKIT `agent_eval` with a fresh-execution outcome oracle |
| `src/selfheal_sql/metrics.py` | Before/after report (accuracy, heal/regression rate, McNemar) |
| `src/selfheal_sql/launch_job.py` | Runs any of the above on HF Jobs GPUs |
| `src/selfheal_sql/build_hf_dataset.py` | Packages splits + SQLite DBs to the HF dataset repo |
| `third_party/agenttune_gate/` | AgentTune's DeploymentGate vendored verbatim (provenance inside) |
| `writeup/` | HF Article |

## Hugging Face artifacts (all public)

- Project page + this source: Space `HideIron/selfheal-sql`
- Dataset (splits + 149 Spider DBs): `HideIron/selfheal-sql-spider`
- Every run log, heal pairs, gate decisions, metrics, audits: `HideIron/selfheal-sql-runs`
- Adapters: `HideIron/tiny-aya-global-sql-selfheal` (DPO, blocked),
  `...-bco` (blocked), `...-bco-replay` (deployed)

## Reproduce

```bash
# 1. splits (needs the official Spider download)
python src/selfheal_sql/make_splits.py <spider_data_dir> data/splits 600 100 500 0
python src/selfheal_sql/make_gate_large.py --total 400 --seed 1
# 2. baseline on an HF Jobs A10G
python src/selfheal_sql/launch_job.py --name baseline-pool -- student.py --split pool
# 3. heal (local; needs HF_TOKEN for the Aya-32B router calls)
python src/selfheal_sql/heal.py --split-file data/splits/pool.jsonl --out data/heal
# 4. retrain (AgentTune on an A10G); best recipe = BCO + success replay
python src/selfheal_sql/launch_job.py --name bco-replay --image python:3.12 \
    --pip "agenttune huggingface_hub" -- train_dpo.py --algorithm bco \
    --replay-run baseline-pool600 --adapter-repo <you>/tiny-aya-sql-bco-replay
# 5. healed evals, gate, metrics
python src/selfheal_sql/launch_job.py --name healed-gate -- student.py --split gate_large \
    --adapter <you>/tiny-aya-sql-bco-replay
python src/selfheal_sql/gate_decision.py --old baseline-gate-large --new healed-gate
python src/selfheal_sql/metrics.py --pair test:baseline-test:healed-test
python src/selfheal_sql/audit_eval.py --run healed-test --split-file data/splits/test.jsonl
```

Models: Tiny Aya Global (student) and Aya Expanse 32B (healer), both open-weights
Cohere Labs models from the hackathon's allowed list. Spider is CC BY-SA 4.0.
AgentTune/AuditKIT are Lexsi Labs (their respective licenses).
