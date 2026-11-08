# Labelling rules

**Status: DRAFT until approved by the author.** Once approved, this file is frozen together with
the test set (`docs/PREREGISTRATION.md` section 3). Any later change goes in the report's
deviations section. The schema is defined in `docs/PREREGISTRATION.md` section 2.

The question for every passage: **does it state a measurable requirement, a limit or permission
that a design, test result or measurement can be checked against?** If not, the label is
`{"requirements": []}`.

## 1. What is a requirement

A requirement has a **parameter**, a **comparator**, one or two **numbers**, and (usually) a
**unit**. Label it when the passage says a measured or designed quantity:

| Wording | Comparator |
|---|---|
| at least, a minimum of, not less than, not lower than | `>=` |
| at most, a maximum of, shall not exceed, no more than, not higher than | `<=` |
| more than, greater than, higher than (as the rule itself) | `>` |
| less than, lower than (as the rule itself) | `<` |
| shall be / is exactly 12 mm, equal to | `==` |
| between 800 mm and 1 000 mm, from 5 to 10 s | `between` (`value` lower, `value_max` upper) |

Rules of thumb:
1. **Non-modal limits count.** Specific-case tables often say "does not exceed 0,05 Ohm" without
   "shall". Label them.
2. **Counts are requirements** when they are a count of things the design must have ("at least two
   wheelchair spaces"): `unit` is null.
3. **One requirement per number-bearing rule.** "500 mm wide and 700 mm long" is two requirements
   (width, length). "Between 800 mm and 1 000 mm" is one.
4. **Same rule, several parameters:** one requirement each ("minimum load of 80 kg" and "force not
   exceeding 25 N" in one passage are two).
5. **A permission is its own requirement** ("may be raised up to 89 dB", "it is permitted to
   reduce this value to 270 mm"): comparator `<=` for "raised up to", `>=` for "reduced to", and
   the case that allows it goes in `conditions`. Do not fold it into the general rule.
6. **Table rows.** If each row pairs a case with a limit, label one requirement per row, with the
   row's case in `conditions`. If a table has more than 8 labelled rows, the passage is dropped
   from sampling, not partly labelled.
7. **Boundary words count as written.** "lower than 250 km/h" is `<` 250, never `<=`.

## 2. What is not a requirement (label empty, or leave out)

1. **Scope sentences:** "This specification applies to all tunnels of more than 1 km in length."
   The 1 km is a condition for other rules in the passage, not a requirement.
2. **Definitions:** "A 'quieter route' means ... a minimum length of 20 km". A number inside a
   definition is not a requirement to be met.
3. **Dates, years, deadlines, versions, article, point and index numbers.**
4. **Reference data and calculation tables:** wavelength tables, roughness coefficients,
   frequency lists with no limit stated.
5. **Procedural thresholds that are not design values:** "diverges ... by more than 25 %" in
   rules about how a Member State calculates something. Test: could a train, tunnel or station be
   checked against it? If it only governs paperwork or a calculation method, it is not labelled.
6. **A bare reference** ("in accordance with the specification referenced in Appendix A") with no
   number of its own.
7. **An exemption without a value** ("compliance is not mandatory at renewal"): no numeric value,
   so no requirement. If it names a numeric limit, label that limit.

## 3. Fields

- **`parameter`**: a short phrase **copied from the passage** (3 to 8 words), naming what is
  limited ("usable surface height", "force to stow the table"). Copy, do not paraphrase.
- **`value` / `value_max`**: the number as stated, read like this: `1 000` is 1000, `0,5` is 0.5.
  Do **not** convert units or round (a limit in cm stays in cm).
- **`unit`**: as written in the passage ("mm", "km/h", "dB", "m/s²", "A"). Null only for plain counts
  ("at least two spaces"). "%" is a unit.
- **`conditions`**: phrases **copied from the passage** that limit when the rule applies ("for
  trains with a design speed lower than 250 km/h", "if a baby nappy changing table is provided",
  "in the lowered position"). One element per condition. Leave out text that only says how
  something is measured. A scope sentence for the whole passage goes into `conditions` of every
  requirement it governs.
- **`quote`**: the shortest stretch of the passage, **verbatim**, that states the rule and
  contains its number(s). Parts may be joined by " ... " only if both parts are verbatim.

## 4. Hard cases and decisions

| Case | Decision |
|---|---|
| Rule in a "Specific case" annex row for one country | Label as normal; the country goes in `conditions` |
| A "shall" with a number that is a reference ("point 4.2.2.9") | Not a number, ignore |
| Number in a figure caption or footnote | Not labelled |
| Same rule repeated in two sentences of one passage | One requirement |
| Alternative values ("1 435 mm or 1 668 mm") | One `==` requirement per value |
| "Not less than X and not more than Y" | `between` |
| Rule whose number is spelled out ("two spaces") | `value` 2, unit null |
| Passage is cut mid-rule | Label only what is complete in the passage |
| Unsure whether it is a requirement | Leave it out and tick `unsure` in the tool |

Every `unsure` passage is reviewed once by the author after the first pass and then decided by
the rule above; the final label is never "unsure". The count of `unsure` decisions per split is
reported.

## 5. Process

1. **Drafts** for train and validation may come from an LLM that is not a compared system.
   Gemini is excluded; the draft model and prompt are recorded in the repo.
2. **Test passages** are labelled or fully checked by hand by the author. Drafts for the test set
   are made in an isolated session; test text and labels are never loaded into any session that
   writes prompts or tunes models.
3. **Time** is logged per split by the labelling tool (start/stop per session).
4. **Consistency:** 15 test passages are re-labelled at least 3 days after the first pass; the
   agreement score is reported as a single-reviewer consistency figure.
5. **Freeze:** the test files are hashed (SHA-256) and the hashes committed before any training
   or prompt tuning.
