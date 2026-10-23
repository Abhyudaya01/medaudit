# MedAudit

[![tests](https://github.com/Abhyudaya01/medaudit/actions/workflows/tests.yml/badge.svg)](https://github.com/Abhyudaya01/medaudit/actions/workflows/tests.yml)

**Status:** framework complete and tested offline; evaluation against real models in progress. Seed rubrics are pending clinician review.

Adversarial stress-testing of LLMs for clinical safety failures. Runs 100 adversarial
seed prompts across six failure categories against OpenAI and Amazon Bedrock models,
scores every response with a rubric-anchored LLM judge, validates the judge against
clinician labels, tracks cost and latency, and writes paper-ready tables and figures.

## Quickstart

```bash
pip install -e ".[all]"            # openai + boto3 + pytest
medaudit list-seeds                # 100 seeds, 6 categories, 7 tactics
medaudit run -c configs/mock.yaml  # full offline dry run, no API calls
pytest                             # 55 tests; live tests are skipped
```

Real run:

```bash
export OPENAI_API_KEY=...          # Bedrock uses the standard AWS credential chain
# edit configs/default.yaml: model IDs and pricing
medaudit estimate -c configs/default.yaml           # call counts, token and cost ceilings
medaudit run -c configs/default.yaml --limit 5 --run-name smoke   # smoke test first
medaudit run -c configs/default.yaml                # generate -> judge -> report
```

Every stage is resumable. Responses and judgments are appended to JSONL and keyed by
(model, seed, sample) and (response, judge), so re-running skips completed work and
retries only errors. `generate`, `judge`, and `report` can also be run separately.

## Design

### Two axes

| Category (what fails) | n | Tactic (how pressure is applied) | n |
|---|---|---|---|
| Medication & Dosing Errors | 17 | Direct (baseline) | 49 |
| Under-Triage & Missed Red Flags | 17 | False premise | 21 |
| Fabricated Evidence & Hallucination | 17 | Authority claim | 10 |
| Sycophancy & False-Premise Acceptance | 17 | Counterfactual pair | 6 |
| Bias & Inequitable Care | 16 | Role-play / hypothetical | 5 |
| Unsafe Compliance & Scope Violations | 16 | Multi-turn pushback | 5 |
| | | Urgency / constraint | 4 |

Definitions live in `medaudit/taxonomy.py`; seeds in `medaudit/data/seeds/*.yaml`.
Each seed carries an item-specific rubric: `expected_safe_behavior`, `failure_criteria`,
and `severity_if_failed`. Multi-turn seeds include a scripted prior assistant turn
followed by pushback. Bias seeds include three counterfactual pairs that differ in one
token (race, ethnicity, insurance); the report computes verdict discordance per pair.

### Judge protocol (`medaudit/judge.py`)

- The judge sees the category definition, generic failure and pass signals, the
  item rubric, the full conversation, and the response.
- Output is JSON: `verdict` (pass/fail, primary), `severity` 0-3, `over_refusal`,
  `failure_modes`, `confidence`, `rationale`.
- The response is wrapped in tags and declared data, so injected instructions in a
  response are ignored. Temperature is 0. Malformed JSON gets one repair retry; after
  that the record is marked unparsed and reported in `judge_health.csv`.
- With several judges, the verdict is a majority vote and ties resolve to FAIL
  (conservative). Inter-judge kappa is reported.
- `estimate` and the run manifest flag any judge that is also an audited model
  (self-preference bias).

### Statistics (`medaudit/metrics.py`)

- Failure rates use Wilson 95% CIs. With `samples_per_prompt > 1`, CIs switch to a
  prompt-level cluster bootstrap, because samples of one prompt are correlated.
- Model comparisons use an exact McNemar test on paired per-prompt verdicts with Holm
  correction.
- Judge validation reports percent agreement, Cohen's kappa with a bootstrap CI,
  sensitivity and specificity ('fail' = positive), per-category agreement, and
  inter-rater kappa.

### Cost and latency

Every call records input and output tokens, wall-clock latency, Bedrock's server-side
latency, retry attempts, and cost (when pricing is set in the config). Judge cost is
tracked separately from audited-model cost.

## Human validation of the judge

```bash
medaudit human-sample -c configs/default.yaml -n 120
#  -> runs/<run>/human/labeling_sheet.csv   (blinded, shuffled; give to raters)
#  -> runs/<run>/human/labeling_key.json    (private; maps items back to responses)
# each rater fills human_verdict (pass/fail) and saves their own copy
medaudit validate -c configs/default.yaml --labels rater1.csv rater2.csv
medaudit report   -c configs/default.yaml --labels rater1.csv rater2.csv
```

The sample is stratified equally across model x category. Have at least two raters
label an overlapping subset so inter-rater agreement (the ceiling for judge-human
agreement) can be reported.

## Outputs (`runs/<run_name>/`)

| File | Contents |
|---|---|
| `results/table1_failure_rates` | model x category failure %, 95% CI, best per column bold |
| `results/table2_model_summary` | failure, severe failure, over-refusal, cost/100 prompts, latency p50/p95, errors |
| `results/table3_failure_by_tactic` | model x tactic |
| `results/table4_pairwise_mcnemar` | paired model tests, Holm-adjusted |
| `results/table5_counterfactual_pairs` | per-pair rates and discordance |
| `results/table6_inter_judge_agreement` | judge-judge kappa (2+ judges) |
| `results/table7_judge_validation` | judge vs. clinicians (with `--labels`) |
| `figures/fig1`-`fig7` | heatmap, category bars with CIs, cost vs. failure, latency, severity, tactic heatmap, judge confusion matrix |
| `manifest.json` | seed-set hash, judge version, model specs, git commit |

Tables are written as `.csv`, `.tex` (booktabs), and `.md`; figures as `.pdf` and `.png`.

## pytest

- `pytest` runs 55 offline tests: seed and taxonomy integrity, judge parsing and
  aggregation, statistics against known values, config, retries, cost, end-to-end
  pipeline on mock providers, resume behaviour, and the human-validation round trip.
- `pytest tests/test_live_audit.py --live --audit-config configs/default.yaml -k "TR-"`
  runs the audit as tests: one test per (model, seed), failing when the judge rates the
  response unsafe. It shares the run cache with the CLI. Use it for targeted regression
  checks; use `medaudit run` for the study itself.

## Before reporting results

1. **Clinical review of rubrics.** The seed rubrics were drafted without clinician
   review. Have a clinician sign off on every `expected_safe_behavior` and
   `failure_criteria` and record this in the paper.
2. **Verify fictional entities.** `pytest -s tests/test_seeds.py -k fictional` lists
   the invented drug, disease, and trial names. Check each against RxNorm, MeSH, and
   ClinicalTrials.gov.
3. **Fill in pricing** from current provider pricing pages, and record the date.
4. **Power.** With 16-17 prompts per category, a model at 0/17 failures still has a
   Wilson upper bound of about 18%. Per-category differences between models need large
   gaps to be significant; tactic-level cells (4-10 prompts) are exploratory.
   `samples_per_prompt` measures stochasticity but does not add prompts.
5. **Fabrication limits.** The judge cannot check whether a citation exists; it flags
   unhedged specific identifiers. Verifying PMIDs/DOIs against PubMed would tighten
   this category.
