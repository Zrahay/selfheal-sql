"""Stage and upload the splits + the SQLite databases they use to a private HF dataset repo.

Rows are rewritten so db_path is RELATIVE to the dataset root
(databases/<db_id>/<db_id>.sqlite): the same rows then work on the laptop and inside an
HF Job that mounts the repo.

Usage: python build_hf_dataset.py <repo_id> [--no-upload]
Token is read from the HF_TOKEN environment variable (never from a file).
"""

from __future__ import annotations

import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
SPLITS = os.path.join(ROOT, "data", "splits")
STAGE = os.path.join(ROOT, "data", "hf_dataset")

CARD = """---
license: cc-by-sa-4.0
task_categories:
- text-generation
language:
- en
tags:
- text-to-sql
- spider
- self-healing
pretty_name: Self-heal SQL - Spider splits
---

# Self-heal SQL: database-disjoint Spider splits

Splits and SQLite databases used for a self-healing text-to-SQL agent
(detect failures -> classify -> generate corrective examples -> DPO retrain -> gate).

Derived from **Spider** (Yu et al., 2018, https://yale-lily.github.io/spider), CC BY-SA 4.0.

| split | source | purpose |
|---|---|---|
| `pool` | Spider train, DB group A | failure pool; its failures become training data |
| `gate` | Spider train, DB group B | held-out deployment gate; never used to build training data |
| `test` | Spider dev | headline before/after numbers |

No database appears in more than one split (asserted at build time). Rows whose gold query
errors, times out or returns zero rows were dropped.

Row fields: `qid, split, db_id, question, gold_sql, db_path, schema`.
`db_path` is relative to the repo root.
"""


def main() -> None:
    repo_id = sys.argv[1]
    upload = "--no-upload" not in sys.argv

    if os.path.isdir(STAGE):
        shutil.rmtree(STAGE)
    os.makedirs(os.path.join(STAGE, "splits"))

    used: dict[str, str] = {}
    for name in ("pool", "gate", "test"):
        rows = []
        with open(os.path.join(SPLITS, f"{name}.jsonl"), encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                used[r["db_id"]] = r["db_path"]
                r["db_path"] = f"databases/{r['db_id']}/{r['db_id']}.sqlite"
                rows.append(r)
        with open(os.path.join(STAGE, "splits", f"{name}.jsonl"), "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{name}: {len(rows)} rows")

    total = 0
    for db_id, src in used.items():
        dst_dir = os.path.join(STAGE, "databases", db_id)
        os.makedirs(dst_dir)
        dst = os.path.join(dst_dir, f"{db_id}.sqlite")
        shutil.copyfile(src, dst)
        total += os.path.getsize(dst)
    print(f"{len(used)} databases, {total / 2**20:.1f} MiB")

    with open(os.path.join(STAGE, "README.md"), "w", encoding="utf-8") as f:
        f.write(CARD)

    if not upload:
        print("staged at", STAGE, "(not uploaded)")
        return

    from huggingface_hub import HfApi

    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)
    api.upload_folder(repo_id=repo_id, repo_type="dataset", folder_path=STAGE,
                      commit_message="Spider-derived DB-disjoint splits + databases")
    print("uploaded: https://huggingface.co/datasets/" + repo_id, "(private)")


if __name__ == "__main__":
    main()
