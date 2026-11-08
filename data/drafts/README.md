# Draft labels (train and validation only)

Drafts are suggestions shown in the labelling tool. They are not labels until the author has
reviewed them and saved them to `data/labels/`.

- **Drafter:** Claude Sonnet 5.5, in a Claude Code session on 2026-11-08, reading each passage
  with `python -m rail_extract.drafting show` and following `docs/LABELLING.md`. No separate
  prompt file exists; the rules file is the prompt.
- **Checks before saving:** every quote verbatim, every parameter and condition copied, every
  value and unit present in the passage (`python -m rail_extract.drafting save`). Judgement calls
  carry `unsure: true` and a `note`.
- **Not drafted here:** the main test and held-out splits. The drafting helper refuses them, and
  their text was not read in the drafting session.
- **Not a compared system:** the drafter is neither Granite nor Gemini. Any influence of the
  drafter's phrasing on the labels is a limitation to report.
