"""DPO retrain via AgentTune's closed-loop retrain path (run inside an HF GPU Job).

Consumes the healer's verified pairs, builds agenttune TrainingExample objects, and
trains an adapter-only LoRA through agenttune's own
  examples_to_dpo_dataset -> build_retrainer (create_agentic_trainer) -> train()
exactly as the closed loop's run_retrain does (we call build_retrainer so we can save
the adapter explicitly afterwards).

Train/eval consistency: the student evaluates with the Cohere chat template
(student.build_prompt). examples_to_dpo_dataset flattens prompt messages as
"role: content", BUT with role="" it emits content verbatim - so we pre-apply the
chat template here and hand it the final string. Training sees exactly the text the
model sees at inference.

Launched by launch_job.py, which injects --data/--out/--push-repo/--push-path.
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)  # injected by launcher; unused
    ap.add_argument("--out", required=True)
    ap.add_argument("--push-repo", default=None)
    ap.add_argument("--push-path", default=None)
    ap.add_argument("--pairs-repo", default="HideIron/selfheal-sql-runs")
    ap.add_argument("--pairs-path", default="heal/baseline-pool600/pairs.jsonl")
    ap.add_argument("--model", default="CohereLabs/tiny-aya-global")
    ap.add_argument("--adapter-repo", default="HideIron/tiny-aya-global-sql-selfheal")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=None, help="smoke-test cap")
    ap.add_argument("--max-prompt-len", type=int, default=1408)
    ap.add_argument("--rpo-alpha", type=float, default=0.0)
    ap.add_argument("--algorithm", default="dpo", choices=["dpo", "bco"])
    ap.add_argument("--replay-run", default=None,
                    help="BCO only: student run whose CORRECT answers are added as "
                         "positive (label=True) replay rows against forgetting")
    ap.add_argument("--no-push-adapter", action="store_true")
    args = ap.parse_args()

    from huggingface_hub import HfApi, hf_hub_download
    from transformers import AutoTokenizer

    from agenttune.decide.closed_loop.contracts import TrainingExample
    from agenttune.decide.closed_loop.retrain_config import RetrainConfig, build_retrainer

    pp = hf_hub_download(args.pairs_repo, args.pairs_path, repo_type="dataset")
    pairs = [json.loads(l) for l in open(pp, encoding="utf-8")]
    healed = [p for p in pairs if p.get("healed") and p.get("chosen_sql")]
    print(f"[dpo] {len(healed)} verified pairs of {len(pairs)} failures", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)

    def templ(user_content: str) -> str:
        return tok.apply_chat_template(
            [{"role": "user", "content": user_content}],
            add_generation_prompt=True, tokenize=False,
        )

    examples = [
        TrainingExample(
            trajectory_id=p["qid"],
            original_failure_type=p["status_before"],
            root_cause=p["root_cause"],
            # role "" -> _messages_to_text emits content verbatim (no "user: " prefix)
            prompt=[{"role": "", "content": templ(p["prompt_user_content"])}],
            chosen=[{"role": "", "content": p["chosen_sql"]}],
            rejected=[{"role": "", "content": p["rejected_sql"]}],
        )
        for p in healed
    ]

    if args.replay_run:
        assert args.algorithm == "bco", "success replay needs BCO's independent rows"
        import re as _re

        def _extract(t):  # minimal extract_sql (fences + first statement)
            t = _re.sub(r"<think>.*?</think>", "", t, flags=_re.S)
            m = _re.search(r"```(?:sql)?\s*(.*?)```", t, _re.S | _re.I)
            if m:
                t = m.group(1)
            m = _re.search(r"\b(select|with)\b", t, _re.I)
            return t[m.start():].split(";")[0].strip() if m else t.strip()

        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from student import PROMPT  # the exact student prompt builder

        split_rows = {}
        sp = os.path.join(args.data or "/data", "splits", "pool.jsonl")
        for s in map(json.loads, open(sp, encoding="utf-8")):
            split_rows[s["qid"]] = s
        rp2 = hf_hub_download(args.pairs_repo, f"runs/{args.replay_run}/results.jsonl",
                              repo_type="dataset")
        n0 = len(examples)
        for r in map(json.loads, open(rp2, encoding="utf-8")):
            if r["status"] == "correct" and r["qid"] in split_rows:
                s = split_rows[r["qid"]]
                examples.append(TrainingExample(
                    trajectory_id=r["qid"] + "-replay",
                    original_failure_type="success_replay", root_cause="none",
                    prompt=[{"role": "", "content": templ(
                        PROMPT.format(schema=s["schema"], question=s["question"]))}],
                    chosen=[{"role": "", "content": _extract(r["raw_output"])}],
                ))
        print(f"[dpo] +{len(examples)-n0} success-replay rows (label=True)", flush=True)

    cfg = RetrainConfig(
        model=args.model,
        algorithm=args.algorithm,
        output_dir=os.path.join(args.out, "adapter"),
        lora_r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        lora_dropout=0.05,
        lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                             "gate_proj", "up_proj", "down_proj"],
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        beta=args.beta,
        max_steps=args.max_steps,
        extra={
            "bf16": True,
            "logging_steps": 2,
            "save_strategy": "no",
            "report_to": "none",
            "gradient_checkpointing": True,
        },
    )
    if args.algorithm == "dpo":
        # The BCO wrapper instantiates the model itself and rejects model_init_kwargs.
        cfg.extra["model_init_kwargs"] = {"dtype": "bfloat16"}

    # Cohere vocab is ~256k: full-sequence fp32 logits are ~4 GiB per copy and OOM a
    # 24 GB card. use_logits_to_keep restricts logits to completion tokens (~60 here).
    # If this TRL lacks it, shrink sequence caps hard instead.
    import inspect

    import trl

    cfg_keys = set(inspect.signature(trl.DPOConfig.__init__).parameters)
    if "use_logits_to_keep" in cfg_keys:
        cfg.extra.update({"use_logits_to_keep": True,
                          "max_prompt_length": 2944, "max_length": 3072})
        print("[dpo] use_logits_to_keep=True (trl", trl.__version__, ")", flush=True)
    else:
        cfg.extra.update({"max_prompt_length": args.max_prompt_len,
                          "max_length": args.max_prompt_len + 128})
        print("[dpo] trl lacks use_logits_to_keep -> caps", args.max_prompt_len, flush=True)
    # rpo_alpha adds an NLL term on the chosen completion (DPO+SFT hybrid): the model
    # must LEARN to produce the verified fix, not merely prefer it. v1 (pure DPO,
    # 30 steps) failed to move pool accuracy at all - this is the main v2 fix.
    if args.rpo_alpha and "rpo_alpha" in cfg_keys:
        cfg.extra["rpo_alpha"] = args.rpo_alpha
        print("[dpo] rpo_alpha =", args.rpo_alpha, flush=True)
    elif args.rpo_alpha:
        print("[dpo] WARNING: trl lacks rpo_alpha; pure DPO", flush=True)

    lens = sorted(len(tok(templ(p["prompt_user_content"]))["input_ids"]) for p in healed)
    print(f"[dpo] prompt tokens: median={lens[len(lens)//2]} p90={lens[int(len(lens)*.9)]} "
          f"max={lens[-1]} | cap={cfg.extra['max_prompt_length']}", flush=True)

    trainer = build_retrainer(examples, cfg)
    stats = trainer.train()
    print("[dpo] train() returned:", str(stats)[:500], flush=True)

    # Save the adapter explicitly (find the underlying HF trainer/model).
    os.makedirs(cfg.output_dir, exist_ok=True)
    inner = None
    for attr in ("trainer", "_trainer", "dpo_trainer"):
        inner = getattr(trainer, attr, None)
        if inner is not None and hasattr(inner, "model"):
            break
    model_obj = getattr(inner, "model", None) or getattr(trainer, "model", None)
    if model_obj is None:
        raise RuntimeError(
            f"cannot locate trained model on {type(trainer).__name__}; attrs: "
            + ", ".join(a for a in dir(trainer) if not a.startswith("__"))[:800]
        )
    model_obj.save_pretrained(cfg.output_dir)  # PeftModel -> adapter only
    print("[dpo] adapter saved:", os.listdir(cfg.output_dir), flush=True)

    with open(os.path.join(args.out, "train_stats.json"), "w", encoding="utf-8") as f:
        json.dump({"n_pairs": len(healed), "stats": str(stats)[:2000],
                   "config": {k: v for k, v in vars(args).items()}}, f, indent=2)

    api = HfApi()
    if not args.no_push_adapter:
        api.create_repo(args.adapter_repo, private=True, exist_ok=True)
        api.upload_folder(repo_id=args.adapter_repo, folder_path=cfg.output_dir,
                          commit_message=f"DPO LoRA ({len(healed)} self-heal pairs)")
        print("[dpo] adapter pushed:", args.adapter_repo, flush=True)
    if args.push_repo:
        api.upload_folder(repo_id=args.push_repo, repo_type="dataset",
                          folder_path=args.out, path_in_repo=args.push_path or "dpo",
                          commit_message="dpo training stats")


if __name__ == "__main__":
    main()
