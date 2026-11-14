"""Enlarge the held-out gate split (100 -> N) so small accuracy differences are measurable.

The original gate has 100 questions (std. error ~ +/-5pp), too noisy to tell a 2pp
regression from chance. This keeps the original 100 rows (same qids, so old results stay
comparable) and adds more questions drawn from the SAME databases the gate already uses
(Spider-train DB group B), so:
  * no database appears in pool or test (asserted),
  * every database is already in the HF dataset repo (no DB upload needed),
  * gold queries run and return >=1 row (same rule as make_splits.py).

Usage (from repo root):
  python src/selfheal_sql/make_gate_large.py [--total 400] [--seed 1]
Writes data/splits/gate_large.jsonl (absolute db_path, like the other local splits) and
data/hf_dataset/splits/gate_large.jsonl (relative db_path, ready to upload).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from execcheck import check, schema_text  # noqa: E402
from make_splits import usable  # noqa: E402

ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
SPIDER = os.path.join(ROOT, "data", "spider", "spider_data")
SPLITS = os.path.join(ROOT, "data", "splits")
STAGE = os.path.join(ROOT, "data", "hf_dataset", "splits")


def read(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--total", type=int, default=400)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    pool, gate, test = (read(os.path.join(SPLITS, f"{n}.jsonl")) for n in ("pool", "gate", "test"))
    gate_dbs = {r["db_id"] for r in gate}
    seen = {(r["db_id"], r["question"]) for r in gate}

    with open(os.path.join(SPIDER, "train_spider.json"), encoding="utf-8") as f:
        train = json.load(f)
    # Candidates: train questions on the gate's databases, not already in the gate.
    cands = [x for x in train if x["db_id"] in gate_dbs and (x["db_id"], x["question"]) not in seen]
    random.Random(args.seed).shuffle(cands)

    rows = [dict(r, split="gate_large") for r in gate]  # keep original 100, same qids
    schemas = {r["db_id"]: r["schema"] for r in gate}
    n_keep = len(rows)
    for it in cands:
        if len(rows) >= args.total:
            break
        db_id = it["db_id"]
        db_path = os.path.join(SPIDER, "database", db_id, f"{db_id}.sqlite")
        if not os.path.exists(db_path) or not usable(db_path, it["query"]):
            continue
        schemas.setdefault(db_id, schema_text(db_path))
        rows.append({
            "qid": f"gate_{len(rows):04d}", "split": "gate_large", "db_id": db_id,
            "question": it["question"], "gold_sql": it["query"],
            "db_path": db_path, "schema": schemas[db_id],
        })

    # Hard guarantees, asserted not assumed.
    dbs = {r["db_id"] for r in rows}
    assert dbs == gate_dbs, "gate_large introduced a new database"
    assert not dbs & {r["db_id"] for r in pool}, "gate_large shares a DB with pool"
    assert not dbs & {r["db_id"] for r in test}, "gate_large shares a DB with test"
    assert len({r["qid"] for r in rows}) == len(rows), "duplicate qid"
    assert len({(r["db_id"], r["question"]) for r in rows}) == len(rows), "duplicate question"
    bad = [r["qid"] for r in rows if not check(r["db_path"], r["gold_sql"], r["gold_sql"]).correct]
    print(f"gate_large: {len(rows)} questions ({n_keep} original + {len(rows) - n_keep} new) "
          f"over {len(dbs)} databases; gold-vs-gold failures: {len(bad)}")
    assert not bad, bad[:5]

    os.makedirs(SPLITS, exist_ok=True)
    with open(os.path.join(SPLITS, "gate_large.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    os.makedirs(STAGE, exist_ok=True)
    with open(os.path.join(STAGE, "gate_large.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            r2 = dict(r, db_path=f"databases/{r['db_id']}/{r['db_id']}.sqlite")
            f.write(json.dumps(r2, ensure_ascii=False) + "\n")
    print("wrote data/splits/gate_large.jsonl and data/hf_dataset/splits/gate_large.jsonl")
    print("db-disjoint from pool/test: OK")


if __name__ == "__main__":
    main()
