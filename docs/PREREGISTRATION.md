# Pre-registration: LoRA fine-tuning vs prompting for requirement extraction

**Status: approved by the author on 2026-11-08. Frozen at the commit that adds this file.** Author: João Pedro Cardoso.

**Freeze rule.** This file is committed before the first training run or prompt tuning. It is not edited after that commit. Any later change of plan goes in the "Deviations" section of `REPORT.md`, with the reason and the effect on the results.

**Smoke test passed** on 2026-11-08 (section 12). Constrained decoding uses XGrammar 0.2.8.

---

## 1. Question and hypothesis

**Question.** For extracting measurable requirements from EU railway TSI passages, is a LoRA fine-tuned small model better than prompting, and is it worth the effort?

**Hypothesis (H1).** The fine-tuned small model has a higher field-level micro-F1 than the best prompting configuration of the same base model, by at least 0.10.

**Null outcome.** "Prompting wins" or "cannot be told apart" are valid results and will be reported as such. The experiment is not re-tuned to change the outcome.

## 2. Task and output schema

**Input:** one passage (chunk of at most 250 words) with its section heading.

**Output:** JSON, an empty list if the passage states no measurable requirement.

```json
{"requirements": [{
  "parameter": "short phrase copied from the passage",
  "comparator": ">= | <= | > | < | == | between",
  "value": 800,
  "value_max": 1000,
  "unit": "mm",
  "conditions": ["verbatim applicability phrase"],
  "quote": "verbatim words stating the rule"
}]}
```

- `value_max` is set only for `between`. `unit` is null for plain counts.
- `parameter` and `quote` must be non-empty (`minLength: 1` in the JSON schema).
- A table row is one requirement.
- A permission or exception ("it is permitted to reduce this to 270 mm") is its own requirement, with the case in `conditions`.
- A number that only sets scope ("tunnels of more than 0,5 km") is a condition, not a requirement.
- Dates, cross-references and legal citations are not requirements.

Detailed labelling rules live in `docs/LABELLING.md`, written before labelling starts and frozen with the test set.

## 3. Data

**Corpus.** Nine EUR-Lex consolidated texts, 2,503 chunks (at most 250 words, 40-word overlap only inside over-long paragraphs): the Interoperability Directive and the CCS, PRM, ENE, SRT, NOI, OPE, INF (`02014R1299-20230928`) and LOC&PAS (`02014R1302-20250427`) TSIs.

**Candidate filter.** A heuristic marks a chunk as a candidate if it has a number with a unit and normative wording. It finds 471 of 2,503 chunks. It is used for sampling only and is not a label.

**Groups (leakage control).**
1. Group = section prefix, up to four numeric levels.
2. All specific-case sections of one document form one group.
3. Groups containing any chunk pair with 5-word-shingle Jaccard of 0.5 or more are merged, across documents too.
4. A group goes entirely into one split. Any group linked to INF leaves the training pool.

**Splits.**

| Split | Target size | Source | Sampling (filter match / numeric but no match / no number) |
|---|---|---|---|
| Held-out TSI | 30 | INF only | 24 / 5 / 1 |
| Main test | 70 | Other 8 documents | 54 / 12 / 4 |
| Validation | 50 | Other 8 documents | Same proportions as main test |
| Train | 250 | Other 8 documents | Same proportions, aiming for about 150 positives |

The split script uses a fixed seed (20261108). The filter-miss strata exist so the regex baseline's recall is not inflated by sampling. The true share of positives is reported and test passages are not resampled to reach a target.

**Labelling.**
- An LLM may draft labels. Gemini must not, because it is a compared system.
- João reviews every train and validation label, and labels or fully checks every test label.
- Test drafts are made in an isolated session. Test passages and labels are never loaded into a session that writes prompts or tunes models.
- 15 test passages are re-labelled after at least 3 days. Agreement (same scoring as section 6) is reported as a single-reviewer consistency figure.
- Labelling time is logged per split.

**Freeze.** Test files are hashed (SHA-256) and the hashes committed before any training or prompt tuning. Code reads the test files only with an explicit `--final` flag, and a test asserts no group appears in two splits.

## 4. Systems

Base model: `ibm-granite/granite-4.2-3b`, revision `4e9c275e232f10211ec88644d106a1f6a1b60e06`, Apache 2.0. Fallback: `HuggingFaceTB/SmolLM3-3B`, revision `a07cc9a04f16550a088caea529712d1d335b0ac1`, Apache 2.0. The fallback is used only if the base model fails section 12. If used, it replaces the base model in every small-model system.

| ID | System |
|---|---|
| R0 | Regex and rules, written from the train set only |
| B1 | Base model, zero-shot prompt |
| B2 | Base model, few-shot prompt (examples from train only) |
| B3 | Base model, the better of the B1 and B2 prompts on validation, with schema-constrained decoding |
| L1 | LoRA model, unconstrained |
| L2 | LoRA model, schema-constrained decoding |
| A1 | `gemini-3.5-flash-lite` with schema-enforced output (the default in `rail-compliance-agent`); fallback `openai/gpt-oss-120b` on Groq with strict schema if Gemini's free tier cannot complete the run |

**Selected systems.** Both are chosen on validation only, by field micro-F1, and fixed before the test set is opened.
- **P\*** is the best of B1, B2, B3. On a tie, B3.
- **L\*** is the better of L1 and L2, using the mean across three seeds. On a tie, L2.

**Tuning budget (equal effort, fixed in advance).**
- Prompts: at most 8 variants for the small model, at most 3 for A1, all scored on validation.
- Few-shot: k in {3, 5, 8}. The examples must include one empty case, one conditional and one with several requirements.
- Thinking mode: B1 and B2 may use off or low, chosen on validation. B3, L1 and L2 use it off, so output is JSON only.
- LoRA: at most 8 hyperparameter configurations, run with seed 1, scored on validation.
- Final LoRA configuration: seeds 1, 2 and 3. No seed is selected.
- Generation is greedy for local models and temperature 0 for APIs.

Training uses PEFT with resumable checkpoints on a free GPU. A small overfit test comes first.

## 5. Test procedure

1. Freeze every system, prompt and configuration, then commit.
2. Run all systems once on the test sets. Save raw predictions.
3. Score from saved predictions.

If a bug is found afterwards, it may be fixed only if it is unrelated to any one system's performance (for example, in the scorer). All systems are then re-scored and the fix is reported as a deviation. Nothing is re-tuned after the test set is opened.

Invalid JSON or a schema-invalid output counts as an empty prediction, and is also counted separately.

## 6. Metrics

**Primary metric: field-level micro-F1 on the main test set (70 passages).**

Scoring:
1. **Normalise.** Numbers follow the `numbers_in` logic of `rail-compliance-agent` (`1 000` is 1000, `0,5` is 0.5, number words are read). Units use a small alias map (`m/s2` and `m/s²`). Text fields are lowercased, punctuation removed and stop-words dropped. Alias and stop-word lists are built from train only and frozen with the test set.
2. **Align** predicted and gold requirements within a passage by maximum matched fields (Hungarian). A pair is eligible if value and unit match, or if `parameter` has token Jaccard of 0.5 or more.
3. **Score fields:** `parameter` (Jaccard of 0.5 or more), `comparator` (6-way), `value` (`value` and `value_max` together), `unit`, and each element of `conditions` (Jaccard of 0.5 or more).
   - TP: gold non-null and predicted equal.
   - FP: predicted non-null and wrong, or requirement unmatched.
   - FN: gold non-null and wrong or missing, or requirement unmatched.
   - A wrong non-null value counts as one FP and one FN.
4. `quote` is not in F1.

**Guardrail: number fidelity.** Number-fidelity error for a passage means at least one `value`, `value_max` or `unit` in the output is absent from the passage (heading plus text, after normalisation).

**Secondary metrics.**
- Field precision, recall and F1, overall and per field
- Exact-match rate per passage (including correctly empty outputs)
- Number-fidelity error at value level (share of output values and units not in the passage)
- Valid-JSON and schema-valid rate
- False-extraction rate on passages with no requirement, overall and per sampling stratum
- Verbatim-quote rate
- Coarse comparator accuracy (min, max, exact, range)
- F1 after a deterministic verifier that drops requirements whose numbers are absent from the quote

**Operational measures** (reported once per configuration, on one logged GPU model):
- Training time, peak GPU memory, adapter size
- Inference latency (p50, p95) and throughput at batch sizes 1, 4, 8 and 16, with and without 4-bit quantisation
- Primary metric of the quantised model (secondary only)

**Cost.** Logged labelling hours and engineering hours, GPU hours, API requests and tokens. Real money spent is expected to be €0. Separately, a labelled "shadow" cost uses published paid prices on the run date and a stated hourly rate, to give a per-request cost and a break-even volume.

## 7. Analysis plan

**Bootstrap.** Paired percentile bootstrap: 10,000 resamples of passages with replacement, seed 20261108. Each resample computes both systems' metric and their difference.

**Seeds.** For LoRA systems, per-passage TP, FP and FN counts are averaged over the three seeds before F1 is computed. The mean and SD of the metric across seeds are reported separately.

**Primary comparison:** L\* minus P\*, on field micro-F1, main test set. Result: point estimate Δ and 95% CI [L, U].

**Sensitivity.** The bootstrap is repeated resampling section groups rather than passages. If the two give different outcomes, the more conservative outcome is reported.

**Secondary comparisons** (estimates with 95% CIs, no significance claims, no multiplicity adjustment): L\* vs A1, P\* vs A1, L2 vs L1, B3 vs B2, every LLM vs R0, and the secondary metrics above.

**Held-out TSI (30 passages).** The same estimators, reported separately and as descriptive only. No decision rests on it.

**Expected precision.** With 20% discordant passages, the 95% interval for a paired exact-match difference is about ±10 points at n = 70 and about ±16 points at n = 30. This is an approximation for planning. Differences smaller than this will often not be distinguishable, and the report will say so plainly.

## 8. What counts as worthwhile

The threshold of 0.10 is a judgement. Smaller differences are below what this sample can resolve, and would not repay the extra labelling and training work.

| Outcome | Rule |
|---|---|
| **A. Fine-tuning better and worthwhile** | Δ ≥ 0.10 and L > 0, and the guardrail holds |
| **B. No worthwhile advantage** | U < 0.10 |
| **C. Cannot be told apart** | Anything else, including a failed guardrail with Δ ≥ 0.10 |

**Guardrail.** L\* minus P\* in passage-level number-fidelity error must be at most +0.03 (point estimate). A model that fabricates more numbers does not qualify as better.

If L > 0 under outcome B, the report states "a real but small gain".

## 9. Recommendation rule

The burden of proof is on fine-tuning, because prompting costs less.

1. Recommend **fine-tuning** only if outcome A holds and the CI lower bound of L\* minus A1 on field micro-F1 is above −0.10.
2. Otherwise recommend **prompting**:
   - the small model (P\*) if the CI lower bound of P\* minus A1 is above −0.10;
   - otherwise the API model (A1).
3. Under outcome C, the report says the evidence is inconclusive, and that prompting is the default for cost reasons.

The report may add when the answer would change (volume, privacy, offline use, rate limits), labelled as judgement rather than result.

## 10. Exclusions and limits known in advance

- One domain, English only.
- One reviewer, whose reading is not legal authority. Consolidated texts have no legal effect.
- Small test sets; held-out results are descriptive.
- Figures and images were dropped in parsing. Tables are flattened to text.
- Chunk boundaries can separate a rule from its context.
- Test composition is enriched in candidates, so false-extraction rates do not reflect the natural 19% candidate share of the corpus.
- A1 is non-deterministic and run once. Its model may change behind the same name; the run date and model ID are recorded.

## 11. Reproducibility

- Library versions are locked in `uv.lock`, and model revisions are pinned as above.
- Weights are not committed. Adapters, if published at all, are published only after asking.
- Saved predictions are committed, so a fresh clone reproduces every reported number by script.

## 12. Smoke test before the freeze

This uses no training and no corpus split. It confirms that the base model:
1. loads at the pinned revision;
2. generates with `enable_thinking=False`;
3. produces schema-valid JSON under constrained decoding, using a made-up sentence.

**Result (2026-11-08, Apple M1 Pro, 16 GB, CPU, transformers 5.22.2, XGrammar 0.2.8).** All three passed, on two made-up sentences, including a batch of 2. GPU (MPS) generation on this machine ran out of memory, so CPU was used. The Mac is not used for experiments, and these runs are not performance measurements.

The fallback is used if any of these fails and cannot be fixed by a configuration change within 2 hours. The same fallback rule applies if the overfit test in step 6 fails on the free GPU, or if the model does not fit in its memory. Using the fallback at step 6 is reported as a deviation.
