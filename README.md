# Jev vs frontier LLMs: agent-routing benchmark

Does a dedicated decision model (TypeSafe Jev) beat a frontier LLM with structured output on
enterprise routing decisions, and *where* is the workload boundary? The aim is to find where each
approach makes sense, not to crown a winner.

**Read this file for**: how to run the benchmark, how the code fits together, what the dataset
contains, and how to read the results.
**Read [jev.md](jev.md) for**: the full narrative write-up, the numbers explained in context, and
concrete examples of what each system got wrong.
**Open [results/results.xlsx](results/results.xlsx) for**: every case, every system's answer and
the ground truth in one spreadsheet, no Python required.

Each case is one request. Three decisions are made against the same state:

| Decision | Type | Meaning |
|---|---|---|
| `agent` | choice | trade_surveillance / market_risk / compliance / human |
| `manual_routing_required` | noul (yes/no) | must a person make or approve the routing decision itself? |
| `high_risk` | noul (yes/no) | material regulatory, conduct or market consequence? |

**`manual_routing_required` is not "is this serious."** That's an easy assumption to make from
the name alone, and it's wrong: see jev.md's "What a mistake actually looks like" for a case
where that distinction matters. It means *does the routing decision itself need a
person to make it*: true only when a request explicitly asks for a human, or the evidence is too
incomplete or conflicting to route automatically. A clearly-worded, high-risk case still gets
`manual_routing_required = false` if it routes cleanly, because a human reviewing it downstream (a
surveillance analyst, say) is the normal queue, not a special flag. This is a narrow, defensible
rule and also a judgment call made by one person, see "Dataset" below for more on that.

## Quick start

```bash
uv venv && uv pip install -r requirements.txt     # or: python -m venv .venv && pip install -r requirements.txt
cp .env.example .env                               # add a real OPENROUTER_API_KEY (sk-or-...)
python benchmark.py --smoke                        # 1 case/system, prints raw responses, verify shapes
python benchmark.py --repeats 3 --split            # full run -> results/raw.jsonl (resumable)
python analyze.py                                  # -> results/metrics.json, summary.md, summary.csv
python visuals.py                                  # -> results/figures/*.png
python export_excel.py                             # -> results/results.xlsx
pytest -q
```

`python benchmark.py --dry-run` uses an offline simulator and writes to `results/dryrun/`. Those
numbers are **fake** (a random-noise generator, not a real model) and every chart is watermarked.
It exists only to exercise the pipeline without spending API credits: useful for checking a code
change didn't break anything before running the real thing.

## Understanding the codebase

The pipeline runs in this order; each stage reads only what the previous one wrote, so you can
rerun any stage on its own once its inputs exist:

1. `data/build_dataset.py` writes `data/cases.json`: the 100 cases. Run once, or again to relabel.
2. `benchmark.py` reads `cases.json`, calls Jev and the LLMs, and writes `results/raw.jsonl` and
   `results/run_meta.json` (model IDs, dataset hash, run date). The only stage that spends money.
3. `analyze.py` reads `raw.jsonl`, computes every metric, and writes `results/metrics.json` (the
   programmatic source of truth) plus `results/summary.md`/`summary.csv` (the same numbers,
   human/spreadsheet readable).
4. `visuals.py` and `export_excel.py` both read `metrics.json`/`raw.jsonl` and write
   `results/figures/*.png` and `results/results.xlsx` respectively. Neither makes an API call.

| File | What it does |
|---|---|
| `spec.py` | The three decision questions, in one place. Both Jev's native request and the LLM prompt/JSON-schema are generated from this same dict, so neither system gets a definition or hint the other lacks. Change a question's wording here, not in two places. |
| `backends.py` | Talks to OpenRouter (`HttpTransport`, with retry/backoff) or simulates it (`MockTransport`, for `--dry-run`). One `call_*` function per system-type, one `parse_*` function per response shape. Parsing is separate from calling, and raw responses are saved untouched, so a parser bug can be fixed and rerun without spending API credits again: only `benchmark.py` costs money; everything downstream just reads `raw.jsonl`. |
| `benchmark.py` | The runner. Reads `.env` for model IDs, resumes an interrupted run (skips `(case, arm, repeat)` triples already in `raw.jsonl`), runs calls concurrently, checks the API key and model IDs before spending anything (`preflight`). |
| `baselines.py` | The two non-LLM systems, computed offline (no API calls, so they're free to rerun): hand-written keyword rules, and TF-IDF + logistic regression with 5-fold cross-validation (so a case is always scored by a model that never trained on it). |
| `analyze.py` | Turns raw responses into every metric reported: accuracy (with confidence intervals), accuracy by case category, the human-review confusion matrix, calibration (ECE/Brier/AUROC), and a confidence-routing policy simulation. This is the file to read if you want to know exactly how a number in `jev.md` was computed. |
| `metrics.py` | The underlying statistics (Wilson interval, ECE, Brier score, etc.) as small standalone functions, independent of this benchmark's data shapes: reusable, and unit-tested in `tests/test_metrics.py`. |
| `visuals.py` | Renders the 5 PNG charts from `metrics.json`. No API calls. Every metric name carries an explicit ↑/↓ so "higher/lower is better" is never left to be inferred, and `place_labels()` is a small generic collision-avoider so chart labels never overlap each other or the data. |
| `export_excel.py` | Writes `results/results.xlsx`: see "Understanding the results" below. |
| `tests/` | `test_metrics.py` unit-tests the statistics; `test_export.py` smoke-tests the Excel export. Both run offline, no API calls: `pytest -q`. |

## Understanding the dataset

`data/build_dataset.py` builds `data/cases.json` (run it again after editing to regenerate). It's
a plain Python list of `(category, request, expected_agent, acceptable_agents,
manual_routing_required, high_risk)` tuples plus a docstring explaining the labeling rules: read
that docstring before trusting any label.

**100 cases, 5 categories:**

| Category | n | What it tests |
|---|---|---|
| `clear` | 30 | Unambiguous requests: the floor every system should clear |
| `ambiguous` | 20 | Plausibly routes to more than one agent |
| `missing_evidence` | 15 | Key facts aren't in the request |
| `conflicting_evidence` | 15 | The request itself contains contradictory signals |
| `adversarial` | 20 | Prompt injection, keyword bait, negation, non-English/full-width text, embedded JSON |

**Labels are deliberately not collinear**: a compliance question can be high-risk, a surveillance
alert can be low-risk, a market-risk case can require manual routing. This matters because a
system that just learns "surveillance → high risk" would score well on a collinear dataset
without actually reasoning about each decision independently. Running `python data/build_dataset.py`
prints the actual cross-tabulations (e.g. `agent==human vs manual_routing_required`) so you can
check this holds before trusting results built on it; it also asserts there are no duplicate
requests and that every case's `expected_agent` is one of its own `acceptable_agents`.

**Labels are unreviewed.** One person wrote and labeled every case for this benchmark. Treat that
as a starting point, not an authority: `jev.md`'s "What a mistake actually looks like" section
documents a real case where a reader's pushback found a genuinely contestable label (a report of
confirmed, ongoing spoofing, labeled `manual_routing_required = false` under the narrow "does the
routing decision need a person" rule: a reasonable alternative rule would flip it). If you disagree with
a label, you're probably not the first person to; relabel it in `build_dataset.py`, rerun, and see
what changes. No confidential data of any kind is in this dataset.

## Understanding the results

Everything in `results/` is regenerated by `analyze.py` / `visuals.py` / `export_excel.py` from
`raw.jsonl`: nothing there is hand-edited, so if a number looks wrong, the fix belongs upstream,
not in the output file.

- **`raw.jsonl`**: every API call's raw, unmodified response, one JSON object per line, append-only.
  This is the ground truth for everything else; if you don't trust a computed metric, this is
  where to check it. Never overwritten between runs (resumable).
- **`run_meta.json`**: which model IDs were used, the dataset's content hash (so you can tell if
  results came from a different version of the dataset), and the run date.
- **`metrics.json`**: every computed metric, structured for the charts to consume. The
  programmatic source of truth; `summary.md`/`.csv` and the charts are all derived from this file
  alone, not from `raw.jsonl` directly.
- **`summary.md` / `summary.csv`**: the same metrics as a quick-read table / spreadsheet-ready
  CSV, one row per system.
- **`results.xlsx`**: the full detail, every case, every system's actual answer, and the ground
  truth, in one workbook with a "Read Me" sheet explaining each tab. Start here if you want to
  build your own charts or spot-check specific cases without writing any code. The **Compare**
  sheet is the fastest way to eyeball one case across every system at once (one row per case,
  each system's answer as adjacent columns, green/red filled by correctness).
- **`figures/*.png`**: the 5 charts embedded in `jev.md`:
  1. `1_hero_scorecard.png`: the headline comparison table (accuracy, calibration, escalations, latency, cost)
  2. `2_accuracy_by_category.png`: which system wins which kind of case (outlined = best per column)
  3. `3_reliability.png`: is each system's stated confidence trustworthy? (see the chart's own caption for what ECE means)
  4. `4_confidence_routing.png`: what happens if you raise the bar for auto-approving a case
  5. `5_latency_vs_cost.png`: what a decision actually costs, in time and money

Every chart's caption explains how to read it and defines any statistic it shows (ECE, confidence
intervals, etc.): you shouldn't need this README open to understand a chart on its own.

## Conclusions

The short version, with the important caveats attached: **read [jev.md](jev.md) for the full
argument, the concrete failure examples, and the nuance the numbers alone don't carry**:

- Headline accuracy (all three decisions correct), with 95% intervals from a cluster bootstrap by
  case (not a naive pool of the 300 correlated case-repeat rows): Jev 47% (37-56%), GPT-5.6-sol 57%
  (48-66%), Claude Sonnet 5 60% (50-68%). Those three overlap, but a **paired** bootstrap (same
  cases, both systems) shows Claude's edge over Jev is real (+12.7pp, 95% CI +2.7 to +22.7,
  p=0.019) while GPT's edge over Jev is not clearly resolved (+10.0pp, 95% CI -0.7 to +20.7,
  p=0.072).
- Among the 135 case-repeats that needed manual routing, Jev produced 0 false negatives (1 for
  Claude, 9 for GPT). Among the 165 that didn't, Jev flagged 107 anyway (65%, vs 30% and 18%): the
  most conservative system by a wide margin, and (see the caveat above) that 65% figure is somewhat
  sensitive to one person's definition of `manual_routing_required`.
- Jev had roughly half the calibration error (ECE) of every LLM tested, and was 7-13x faster and
  75-151x cheaper per decision (p50 latency; both ranges span the two LLMs tested).
- Jev's strongest relative results were on missing-evidence and conflicting-evidence cases; the
  LLMs generalized better on clear, ambiguous and adversarial phrasing.
- Splitting one LLM into 3 separate calls instead of one joint call dropped its accuracy from 57%
  to 33% and cost more. That doesn't isolate WHY (the split calls also see less context per call,
  a real confound), but it does show naive call-level decomposition can cost accuracy and money at
  once, which is real evidence for a shared-state decision layer over ad hoc decomposition, not
  proof that Jev is the best implementation of one.

None of this is evidence about any production workload. It's a synthetic, 100-case, single-day
benchmark with unreviewed labels, run to find the shape of the answer, not to settle it.

## Caveats (state these when publishing)

- Synthetic, unreviewed labels; n=100 gives wide confidence intervals. Repeats are pooled, so those
  intervals are somewhat optimistic (three repeats of the same case aren't independent evidence).
- LLM confidence is *verbalised* self-report; Jev's is a model-returned probability from RLCD
  training. Different mechanisms measured on the same 0–1 scale: see `3_reliability.png`'s caption.
- Latency is measured with 4 concurrent workers from one machine and includes network/provider
  variance, not just model compute time.
- Not evidence about any production workload.
