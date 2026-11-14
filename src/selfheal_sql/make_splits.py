"""Build the three DATABASE-DISJOINT splits from Spider.

  pool  (failure pool)  - questions the untrained student attempts; its failures
                          feed the heal loop.  Source: Spider train, DB group A.
  gate  (held-out gate) - DeploymentGate A/B set; never used to make training data.
                          Source: Spider train, DB group B (different DBs from pool).
  test  (final test)    - headline before/after numbers.
                          Source: Spider dev (Spider's own dev DBs never appear in train).

Disjoint DBs mean a "heal" cannot pass by memorising a schema it already saw.

Rows whose gold query errors, times out, or returns ZERO rows are dropped: an empty
gold result would let any empty-result query count as correct.

Usage: python -I make_splits.py <spider_data_dir> <out_dir> [pool gate test seed]
"""

from __future__ import annotations

import json
import os
import random
import sys

from execcheck import check, schema_text  # noqa: E402  (run from this folder)


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def usable(db_path: str, gold_sql: str) -> bool:
    # gold vs gold: must run, and must return at least one row
    import sqlite3

    from execcheck import _connect, _run  # noqa: PLC0415

    try:
        conn = _connect(db_path)
        try:
            rows = _run(conn, gold_sql, 10.0)
        finally:
            conn.close()
    except (sqlite3.Error, Exception):  # noqa: BLE001
        return False
    return len(rows) > 0


def build_rows(items, db_dir, want, rng, tag):
    rng.shuffle(items)
    rows, schemas = [], {}
    for it in items:
        if len(rows) >= want:
            break
        db_id = it["db_id"]
        db_path = os.path.join(db_dir, db_id, f"{db_id}.sqlite")
        if not os.path.exists(db_path) or not usable(db_path, it["query"]):
            continue
        if db_id not in schemas:
            schemas[db_id] = schema_text(db_path)
        rows.append(
            {
                "qid": f"{tag}_{len(rows):04d}",
                "split": tag,
                "db_id": db_id,
                "question": it["question"],
                "gold_sql": it["query"],
                "db_path": db_path,
                "schema": schemas[db_id],
            }
        )
    return rows


def main():
    spider = sys.argv[1]
    out = sys.argv[2]
    n_pool, n_gate, n_test, seed = (
        int(sys.argv[3]) if len(sys.argv) > 3 else 200,
        int(sys.argv[4]) if len(sys.argv) > 4 else 100,
        int(sys.argv[5]) if len(sys.argv) > 5 else 100,
        int(sys.argv[6]) if len(sys.argv) > 6 else 0,
    )
    rng = random.Random(seed)
    db_dir = os.path.join(spider, "database")
    train = load(os.path.join(spider, "train_spider.json"))
    dev = load(os.path.join(spider, "dev.json"))

    # Split TRAIN databases into two disjoint groups (pool vs gate).
    train_dbs = sorted({x["db_id"] for x in train})
    rng.shuffle(train_dbs)
    cut = int(len(train_dbs) * 0.65)
    group_a, group_b = set(train_dbs[:cut]), set(train_dbs[cut:])
    pool_items = [x for x in train if x["db_id"] in group_a]
    gate_items = [x for x in train if x["db_id"] in group_b]

    pool = build_rows(pool_items, db_dir, n_pool, rng, "pool")
    gate = build_rows(gate_items, db_dir, n_gate, rng, "gate")
    test = build_rows(list(dev), db_dir, n_test, rng, "test")

    # Hard guarantees, asserted not assumed.
    dbs = {k: {r["db_id"] for r in v} for k, v in (("pool", pool), ("gate", gate), ("test", test))}
    assert not (dbs["pool"] & dbs["gate"]), "pool/gate share a database"
    assert not (dbs["pool"] & dbs["test"]), "pool/test share a database"
    assert not (dbs["gate"] & dbs["test"]), "gate/test share a database"

    os.makedirs(out, exist_ok=True)
    for name, rows in (("pool", pool), ("gate", gate), ("test", test)):
        with open(os.path.join(out, f"{name}.jsonl"), "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{name}: {len(rows)} questions over {len(dbs[name])} databases")

    # Sanity: gold-vs-gold must be 100% correct under the checker.
    bad = [r["qid"] for rows in (pool, gate, test) for r in rows
           if not check(r["db_path"], r["gold_sql"], r["gold_sql"]).correct]
    print("gold-vs-gold failures:", len(bad), bad[:5])
    print("db-disjoint: OK")


if __name__ == "__main__":
    main()
