"""Student harness: make the model write SQL for a split, execute it, grade it, log it.

One script runs BOTH the baseline (no adapter) and the healed model (--adapter), with the
same prompt, the same greedy decoding and the same parsing, so a before/after difference
comes from training and nothing else.

Usage (inside an HF Job; see launch_job.py):
  python student.py --data /data --split pool --model CohereLabs/tiny-aya-global \
      --out /out/baseline_pool [--adapter path_or_repo] [--limit 20] \
      [--push-repo user/runs-repo --push-path runs/baseline_pool]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from execcheck import check  # noqa: E402

PROMPT = (
    "You are an expert in SQLite. Given the database schema and a question, write ONE SQLite "
    "query that answers the question. Output only the SQL query, with no explanation.\n\n"
    "Database schema:\n{schema}\n\nQuestion: {question}\n\nSQL:"
)


def load_rows(data_root: str, split: str, limit: int | None):
    path = os.path.join(data_root, "splits", f"{split}.jsonl")
    if not os.path.exists(path):  # local layout: <root>/<split>.jsonl
        path = os.path.join(data_root, f"{split}.jsonl")
    rows = [json.loads(line) for line in open(path, encoding="utf-8")]
    return rows[:limit] if limit else rows


def build_prompt(tok, row) -> str:
    msgs = [{"role": "user", "content": PROMPT.format(schema=row["schema"], question=row["question"])}]
    return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dataset root (contains splits/ and databases/)")
    ap.add_argument("--split", required=True)
    ap.add_argument("--model", default="CohereLabs/tiny-aya-global")
    ap.add_argument("--adapter", default=None, help="LoRA adapter path or HF repo id")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--push-repo", default=None)
    ap.add_argument("--push-path", default=None)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = load_rows(args.data, args.split, args.limit)
    print(f"[student] {len(rows)} rows from split={args.split}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda")
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    print("[student] model loaded:", args.model, "adapter:", args.adapter, flush=True)

    prompts = [build_prompt(tok, r) for r in rows]
    order = sorted(range(len(rows)), key=lambda i: len(prompts[i]))  # batch similar lengths
    raw: dict[int, tuple[str, int, int]] = {}
    gen_seconds = 0.0
    for b in range(0, len(order), args.batch_size):
        idx = order[b : b + args.batch_size]
        enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        t0 = time.time()
        with torch.inference_mode():
            out = model.generate(
                **enc, max_new_tokens=args.max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id
            )
        torch.cuda.synchronize()
        gen_seconds += time.time() - t0
        new = out[:, enc["input_ids"].shape[1] :]
        for j, i in enumerate(idx):
            ids = new[j]
            n_new = int((ids != tok.pad_token_id).sum())
            raw[i] = (tok.decode(ids, skip_special_tokens=True), int(enc["attention_mask"][j].sum()), n_new)
        print(f"[student] {min(b + args.batch_size, len(order))}/{len(order)}", flush=True)

    os.makedirs(args.out, exist_ok=True)
    statuses: Counter = Counter()
    n_new_tokens = 0
    with open(os.path.join(args.out, "results.jsonl"), "w", encoding="utf-8") as f:
        for i, r in enumerate(rows):
            text, n_prompt, n_new = raw[i]
            n_new_tokens += n_new
            res = check(os.path.join(args.data, r["db_path"]), text, r["gold_sql"])
            statuses[res.status] += 1
            f.write(
                json.dumps(
                    {
                        "qid": r["qid"], "split": r["split"], "db_id": r["db_id"],
                        "question": r["question"], "gold_sql": r["gold_sql"],
                        "raw_output": text, "status": res.status,
                        "error_message": res.error_message,
                        "pred_rows": res.pred_rows, "gold_rows": res.gold_rows,
                        "prompt_tokens": n_prompt, "new_tokens": n_new,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    n = len(rows)
    summary = {
        "split": args.split, "model": args.model, "adapter": args.adapter, "n": n,
        "execution_accuracy": statuses["correct"] / n,
        "valid_sql_rate": (n - statuses["sql_error"] - statuses["empty_sql"] - statuses["timeout"]) / n,
        "status_counts": dict(statuses),
        "generation_seconds": round(gen_seconds, 2),
        "new_tokens": n_new_tokens,
        "tokens_per_second": round(n_new_tokens / gen_seconds, 1) if gen_seconds else None,
        "max_new_tokens": args.max_new_tokens, "decoding": "greedy",
    }
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("[student] SUMMARY", json.dumps(summary), flush=True)

    if args.push_repo:
        from huggingface_hub import HfApi

        HfApi().upload_folder(
            repo_id=args.push_repo, repo_type="dataset", folder_path=args.out,
            path_in_repo=args.push_path or os.path.basename(args.out.rstrip("/")),
            commit_message=f"student run {args.split}",
        )
        print("[student] pushed to", args.push_repo, args.push_path, flush=True)


if __name__ == "__main__":
    main()
