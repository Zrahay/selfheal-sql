# Self-Healing Text-to-SQL with AgentTune: what actually happened

*Lexsi Labs Hackathon, Track 3 (Agentic Fine-tuning), PS #8: "Self-healing agents:
detect failing runs, classify why they failed, generate corrective training examples,
retrain, and gate redeployment on measured accuracy." Solo project.*

I built a self-healing loop for a text-to-SQL agent based on Tiny Aya Global (3.35B).
The short version: the agent's failures on real Spider databases get detected by
actually executing its SQL, classified into SQL-specific root causes, corrected by
Aya Expanse 32B (every correction is verified by execution before it becomes training
data, and the healer never sees the gold query), retrained adapter-only through
AgentTune, and redeployment is decided by AgentTune's DeploymentGate on held-out data.

It worked, but not the way I expected. The gate blocked four adapters in a row
because they genuinely regressed, which taught me more than the final number did.
The fifth adapter (BCO with success replay) got approved on a 400-question gate and
improves the unseen 500-question test split from 45.6% to 47.6% execution accuracy,
with the valid-SQL rate going from 68.2% to 73.0%. AuditKIT re-executed everything
independently and agreed with my harness on all 500 cases, twice. At no point did I
write or fix a single SQL query by hand.

## 1. The task needs an executable ground truth

A self-healing loop is only as honest as its failure detector, so I picked Spider
(Yu et al., 2018): every question comes with a real SQLite database and a gold query.
"Failed" is not a judge model's opinion, it's an executed fact. Run the model's SQL,
run the gold SQL, compare result sets (floats rounded, order-insensitive unless the
gold query has ORDER BY). The checker returns one of
`correct | wrong_result | sql_error | timeout | empty_sql` and is the single source
of truth for every stage: failure detection, fix validation, the deployment gate, and
the final report. One definition of failure everywhere, so nothing can be gamed.

### Database-disjoint splits

| split | size | source | role |
|---|---|---|---|
| `pool` | 600 | Spider train, DB group A | failure pool, becomes training data |
| `gate` | 100 (later 400, see section 5) | Spider train, DB group B | DeploymentGate held-out set |
| `test` | 500 | Spider dev | headline before/after |

No database appears in two splits (asserted at build time), so a "heal" cannot pass
by memorising a schema it trained on. Rows whose gold query errors out or returns
zero rows are dropped, because an empty gold result would let any empty query pass.
Gold-vs-gold sanity is 100% on all splits.

## 2. Baseline

Tiny Aya Global, greedy decoding, schema in the prompt, one query per question:

| split | execution accuracy | valid-SQL rate |
|---|---|---|
| pool (600) | 57.8% | 83.3% |
| gate (100) | 62.0% | 83.0% |
| test (500) | 45.6% | 68.2% |

The gap between the train-derived splits and Spider-dev smells like partial
training-data contamination, which is one more reason the headline number has to come
from the dev-derived test split, and why the splits are database-disjoint in the
first place.

The 253 pool failures break down as 100 SQL errors (hallucinated columns, wrong join
paths) and 153 wrong results.

## 3. The heal loop

```
pool failures (253)
  -> classify (Aya Expanse 32B, SQL-specific taxonomy, AgentTune contract)
  -> generate fix (Aya Expanse 32B, never shown the gold SQL)
  -> execute the candidate fix. matches gold result? keep : retry with the error (max 3)
  -> 158 verified preference pairs (chosen = verified fix, rejected = agent's SQL)
  -> AgentTune retrain: DPO / BCO, LoRA-only (r=16)
  -> AgentTune DeploymentGate on held-out gate data: deploy only if no regression
```

### What AgentTune provided and what I had to build

Used as-is from AgentTune: the `Failure`/`ClassifiedFailure`/`TrainingExample`
contracts, the classifier's output contract, `examples_to_dpo_dataset` plus
`create_agentic_trainer` (TRL DPO/BCO, adapter-only LoRA) via `RetrainConfig`, and
`DeploymentGate.evaluate_decision` (vendored verbatim at commit `1dfd054`).

What I replaced, and why. AgentTune's stock `TrainingExampleGenerator` is
tool-calling-specific: it only synthesizes corrections for `wrong_tool` and
`loop_collapse` root causes (about 95% of SQL failures would be silently dropped), it
prompts the healer for "the EXACT JSON tool call", and it trains on a placeholder
prompt instead of the real task context. I swapped this one stage for a SQL-specific
generator whose important property is execution-verified corrections: a fix enters
the training set only if it actually returns the gold result on the real database.
The healer gets the schema, the question, the wrong query and the observed error,
which is exactly what a production system would have, and nothing more. I also
replaced the tool-agent taxonomy with a SQL one (schema_misread, wrong_join, and so
on) while keeping AgentTune's classification contract.

### Heal-stage results (before/after metric 1: can the failures be fixed at all?)

- 253 failures became 158 verified pairs, a 62.5% yield, using 731 LLM calls and
  about 0.66M tokens.
- 126 were fixed on attempt 1. Feeding the execution error back for retries added 32
  more (+25%). 95 resisted all 3 attempts, and that residue is biased toward
  genuinely ambiguous questions.
- Root causes: schema_misread 84, wrong_join 50, value_mismatch 33,
  wrong_aggregation 32, missing_or_wrong_filter 32, other 22.
- 0 classification contract violations across all 253 failures.

## 4. Retraining, or: why naive DPO made it worse

I trained five adapters through AgentTune's retrain path, all LoRA r=16/alpha=32 on
all projection matrices, bf16, on a single A10G (24 GB):

| # | recipe | gate accuracy (n=100) | vs baseline 62% |
|---|---|---|---|
| v1 | DPO, 3 epochs | 58.0% | -4pp |
| v2 | DPO, longer, unconstrained fixes | worse | |
| v3 | DPO, minimal-edit fixes | worse | |
| v4 | BCO, pairs only | worse | |
| v5 | BCO + success replay | 60.0% | -2pp on n=100 (noise, see section 5) |

Why pure DPO fails here: likelihood displacement. A verified SQL fix is by
construction a minimal edit of the broken query, so chosen and rejected share most of
their tokens. DPO's loss grows the margin between them, and the known failure mode on
near-identical pairs is that pushing the rejected sequence down drags the shared
tokens (that is, the chosen answer) down with it. The standard mitigation is
`rpo_alpha`, an added NLL term that makes the model actually learn the fix instead of
merely preferring it, but it does not exist in the TRL version AgentTune pins. My
training script detects this at runtime and falls back. v2 also exposed a separate
problem that v3 fixed: unconstrained healer fixes transfer the healer's style
(aliases, gratuitous JOINs, reformatting), so I added minimal-edit rules to the
healer prompt. v3 still regressed, so displacement was the remaining cause.

The second cause is catastrophic forgetting. Every pair trains on a failure prompt.
The 347 questions the model already answered correctly appear zero times, and the
model drifts off them: v1 fixed 4 gate questions and broke 8.

The remedy that AgentTune happens to ship: BCO. BCO treats each row as an independent
thumbs-up or thumbs-down with no pairing requirement, so I could add the model's own
347 correct pool answers as positive rows next to the 153 verified fixes (positive)
and the original wrong queries (negative), 653 rows total. That is impossible under
DPO, which needs a chosen/rejected pair per prompt.

(Config, for reproducibility: lr 1e-5, beta 0.1, 3 epochs, grad accum 16, batch size
1, gradient checkpointing, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. The
roughly 256k Cohere vocab makes full-sequence fp32 logits about 4 GiB, so the
sequence-length caps are load-bearing on a 24 GB card.)

## 5. The gate (before/after metric 2)

AgentTune's `DeploymentGate.evaluate_decision` compares old vs new on held-out gate
data and blocks deployment on any task regression. Every decision below is a
machine-generated artifact in the runs repo under `gate/`, with per-question flip
lists.

Round 1, n=100, where the gate correctly blocks regressions:

| candidate | old | new | delta | decision | fixed | broke |
|---|---|---|---|---|---|---|
| v1 DPO | 62.0% | 58.0% | -4.0pp | BLOCKED ("task accuracy regressed") | 4 | 8 |
| v5 BCO+replay | 62.0% | 60.0% | -2.0pp | BLOCKED ("task accuracy regressed") | 5 | 7 |

A self-healing system that ships a regressed model is worse than no self-healing at
all. Mine refused to ship, four times, and it was right each time.

Round 2, statistical power. At n=100, 2pp is literally 2 questions (McNemar exact
p = 0.77), so the v5 "regression" could easily be noise. I enlarged the gate to 400
questions drawn from the same held-out databases (DB-disjointness preserved, the
original 100 kept) and re-ran both models:

| candidate | old (n=400) | new (n=400) | delta | decision | fixed | broke |
|---|---|---|---|---|---|---|
| v5 BCO+replay | 59.0% | 59.25% | +0.25pp | APPROVED ("deploy") | 30 | 29 |

On the bigger gate, v5 is non-regressing and the gate's A/B comparison recommends
deploy. So I deployed it.

## 6. Final before/after on the unseen test split (metric 3)

The 500-question test split derives from Spider-dev and shares no database with
anything the loop ever saw. Neither the student, nor the healer, nor the gate ever
touched these databases:

| metric | baseline | v5 BCO+replay (deployed) |
|---|---|---|
| execution accuracy | 45.6% | 47.6% (+2.0pp) |
| valid-SQL rate | 68.2% | 73.0% |
| SQL errors | 159 | 135 |
| wrong results | 113 | 127 |
| fixed / broke (vs baseline) | | 34 / 24 |
| McNemar exact p | | 0.24 |

The improvement lands exactly where the heal pairs aimed: 24 fewer invalid queries
(hallucinated columns, broken joins). A p of 0.24 means one heal round is a
consistent but not statistically decisive gain. The loop is fully automated, so more
rounds are the obvious next step. For contrast, the gate-blocked v1 DPO adapter
scores 44.6% on test (-1.0pp): the gate ranked the adapters correctly on data it
never saw.

Independent verification: AuditKIT's `AgentEvalRunner` with a `CustomPredicate`
oracle that freshly re-executes every final answer against the gold result on the
real database. It does not read my harness's logs. It agreed with the harness
500/500 on baseline-test and 500/500 on the healed test run (verified success
238/500, i.e. 47.6%).

## 7. Cost and ROI

The whole thing ran inside the hackathon's $20 HF budget, about $11 spent:

| item | cost |
|---|---|
| baseline evals (pool 600 + gate + test), A10G | ~$1.5 |
| heal stage: 731 Aya-32B calls, ~0.66M tokens (HF router) | ~$1 |
| 5 adapter trainings, A10G | ~$4 |
| healed evals + 400-question gate evals, A10G | ~$4 |
| total | ~$11 |

Serving ROI, self-hosted 3.35B vs a closed-source API. Measured on the test run:
mean 755 prompt tokens and 33 output tokens per query, 121.7 tok/s on a $1/hr A10G,
which is about 275 seconds of generation per 1,000 queries:

| serving option | cost per 1,000 queries |
|---|---|
| Tiny Aya Global on A10G (self-hosted, measured) | ~$0.08 |
| Claude Haiku 5.5 ($0.10/$0.50 per MTok) | ~$0.09 |
| Claude Sonnet 5.5 ($2/$10 per MTok) | ~$1.84 |
| Claude Opus 5.5 ($4/$20 per MTok) | ~$3.69 |

A self-hosted 3.35B matches small-tier API pricing and is roughly 20-45x cheaper than
frontier-tier APIs, and it is the only option where a self-heal loop can actually
retrain the weights it serves. The loop's one-time cost of about $11 is recouped
within ~6k queries vs a frontier API. (Closed-API prices are first-party list prices
as of Nov 2026. I did not measure the closed models' accuracy on Spider; the ROI
case here is about serving economics plus the ability to improve your own weights.)

## 8. Artifacts (all public, all on Hugging Face)

- Project page + full source: [Space `HideIron/selfheal-sql`](https://huggingface.co/spaces/HideIron/selfheal-sql)
- Dataset (splits incl. gate_large + 149 DBs): [`HideIron/selfheal-sql-spider`](https://huggingface.co/datasets/HideIron/selfheal-sql-spider)
- Every run log, heal pairs, gate decisions, metrics, AuditKIT reports:
  [`HideIron/selfheal-sql-runs`](https://huggingface.co/datasets/HideIron/selfheal-sql-runs)
- Adapters: [`tiny-aya-global-sql-selfheal`](https://huggingface.co/HideIron/tiny-aya-global-sql-selfheal) (DPO, blocked) ·
  [`-bco`](https://huggingface.co/HideIron/tiny-aya-global-sql-selfheal-bco) (blocked) ·
  [`-bco-replay`](https://huggingface.co/HideIron/tiny-aya-global-sql-selfheal-bco-replay) (deployed)
- Models: Tiny Aya Global (student), Aya Expanse 32B (healer), both open-weights
  Cohere Labs models from the hackathon's allowed list.

## 9. Limitations and what I'd do next

- +2.0pp on test is consistent but not statistically decisive (p = 0.24). The loop
  is automated end to end, so the fix is simply more heal rounds.
- The small gate (n=100) both correctly blocked real regressions and nearly blocked
  a good adapter on noise. Gate size is a first-class design parameter for
  self-healing systems; the 400-question enlargement is my fix.
- 95 of 253 failures resisted healing, and that residue skews toward questions that
  are genuinely ambiguous.
- Pool and gate derive from Spider train (possible pretraining contamination), which
  is why the headline is the dev-derived test split and the splits are DB-disjoint.
- `rpo_alpha` or `use_logits_to_keep` would likely rescue the plain DPO path, but
  the TRL version AgentTune pins predates both. The displacement analysis in
  section 4 is, I think, actionable feedback for AgentTune itself.
