# Draft labels

Drafts are suggestions. They are not labels until the author has reviewed them in the labelling
tool, except where two drafters agreed (below).

## Drafters
- **a:** Claude Sonnet 5.5. Validation: one Claude Code session, 2026-11-08. Train: fresh
  Sonnet 5.5 subagents, one per slice of 50.
- **b:** Claude Opus 5.5, fresh subagents that never see drafter a's output (instructed not to
  read `data/drafts/`; this is an instruction, not a technical block).
- **Test and held-out:** a single drafter (Opus 5.5, `a` files) in isolated subagents with
  `RAIL_EXTRACT_ISOLATED_DRAFTER=1`. They report only counts back. The author fully checks every
  one of the 100 labels. The main session that writes prompts and code never read those drafts.
- The rules file `docs/LABELLING.md` is the prompt. Neither drafter is Granite or Gemini.

## Checks before a draft is saved
Every quote verbatim, every parameter and condition copied, every value and unit present in the
passage (`python -m rail_extract.drafting save`).

## Agreement protocol (train and validation; fixed before drafter b ran)
A passage goes to the author if the two drafts differ in comparator, value, value_max, unit
(normalised) or number of requirements, or if both drafters marked it unsure. Other passages are
saved as `auto-agreed` labels (drafter a's text). A random 10 % of them (seed 20261108) stay in
the author's queue as an audit of how often agreement hides an error. See `src/rail_extract/agree.py`.

This replaces the plan that the author reviews every train and validation label
(`docs/PREREGISTRATION.md` section 3). It is a deviation and is reported in `REPORT.md`.

## No human review (decision of 2026-11-09)
The author decided not to review labels: they do not have the domain knowledge to validate them.
Consequences, all to be stated in `REPORT.md` as deviations and limitations:
- No label in any split has been checked by a human. The pre-registered "hand-labelled or fully
  checked by the author" test set (`docs/PREREGISTRATION.md` section 3) does not hold.
- Results measure agreement with LLM-drafted labels, not with a verified ground truth.
  Labelling hours are zero for human time; LLM drafting cost is reported instead.
- The 10 % audit of auto-agreed labels is not done, so the error rate of agreement is unknown.
- Passages where the two drafters disagree still need a decision rule (not yet fixed).
