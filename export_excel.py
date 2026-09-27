"""Exports every case, every system's response and the ground truth to one Excel workbook,
so someone without Python can inspect the raw data and build their own pivots/charts.

    python export_excel.py [--dir results]   ->  <dir>/results.xlsx

Reuses analyze.py's raw-response loader and baselines.py's offline classifiers, so the
numbers in this workbook always match results/metrics.json and summary.md - it reads the
same inputs, it just doesn't aggregate them.

Sheets:
  Read Me   - what every column means and the caveats that apply to all of them
  Cases     - the 100 cases and their ground-truth labels (data/cases.json, unrolled)
  Responses - long format: one row per (case, system, repeat). The rawest useful view.
  Compare   - wide format: one row per (case, repeat), each system's answers as columns
              side by side, for eyeballing a single case across every system at once.
  Summary   - the same per-system aggregate table as results/summary.csv.
"""
import argparse
import csv
import json
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

import analyze
import baselines
import spec

HEADER_FILL = PatternFill("solid", fgColor="1F2933")
HEADER_FONT = Font(color="FFFFFF", bold=True)
GOOD_FILL = PatternFill("solid", fgColor="DCFCE7")   # pale green: correct / true
BAD_FILL = PatternFill("solid", fgColor="FEE2E2")    # pale red: incorrect / false
WRAP = Alignment(wrap_text=True, vertical="top")

# Display order and short names, independent of whatever order arms happened to appear in
# raw.jsonl (see the note in analyze.py: completion order is not submission order).
SYSTEM_ORDER = ["jev", "llm", "llm_split", "tfidf_lr", "rules"]


def label_for(system, model):
    short = (model or "").split("/")[-1]
    return {"jev": "Jev", "llm": short, "llm_split": f"{short} (3 calls)",
            "tfidf_lr": "TF-IDF + LR", "rules": "Keyword rules"}.get(system, system)


def add_sheet(wb, name):
    ws = wb.create_sheet(name)
    return ws


def write_header(ws, headers):
    ws.append(headers)
    for col, _ in enumerate(headers, start=1):
        cell = ws.cell(row=1, column=col)
        cell.fill = HEADER_FILL
        cell.font = HEADER_FONT
    ws.freeze_panes = "A2"


def autosize(ws, widths):
    """`widths` is a dict {column_letter_or_index: char_width}; anything unlisted gets a default."""
    for i in range(1, ws.max_column + 1):
        letter = get_column_letter(i)
        ws.column_dimensions[letter].width = widths.get(i, widths.get("default", 14))


def fill_bool(cell, value):
    cell.fill = GOOD_FILL if value else BAD_FILL


# --------------------------------------------------------------------------- Read Me
def write_readme(wb, res_meta):
    ws = add_sheet(wb, "Read Me")
    ws.column_dimensions["A"].width = 30
    ws.column_dimensions["B"].width = 100
    rows = [
        ("Jev vs LLMs vs classic baselines - full data export", ""),
        ("", ""),
        ("What this is", "Every one of the 100 synthetic routing cases, what each system (Jev, two LLMs, "
                          "a TF-IDF classifier, keyword rules) answered for it, and the ground-truth label it "
                          "was scored against. See jev.md in the repo for the write-up these numbers support."),
        ("Run", f"{res_meta.get('date', 'unknown date')}, {res_meta.get('repeats', '?')} repeat(s) per case, "
                f"dataset hash {res_meta.get('dataset_sha256', '?')}"),
        ("", ""),
        ("Sheet: Cases", "The 100 cases and their ground truth. `category` is the difficulty bucket "
                          "(clear/ambiguous/missing_evidence/conflicting_evidence/adversarial). "
                          "`manual_routing_required` and `high_risk` are booleans; see the note on manual_routing_required below."),
        ("Sheet: Responses", "One row per (case, system, repeat): what that system answered, whether each "
                              "of the 3 decisions was correct, its stated confidence, latency and cost. "
                              "This is the rawest view - filter it by `arm` to isolate one system, or by "
                              "`case_id` to see every system's attempt at one case."),
        ("Sheet: Compare", "One row per (case, repeat), every system's answer for `agent` as adjacent "
                            "columns, so you can eyeball a single case across all systems without pivoting. "
                            "Green = correct, red = incorrect, against the Cases sheet's ground truth."),
        ("Sheet: Summary", "The same aggregate per-system metrics table as results/summary.csv "
                            "(accuracy, calibration, latency, cost)."),
        ("", ""),
        ("IMPORTANT: manual_routing_required is not \"is this serious\"", "It means: does the ROUTING decision "
         "itself need a person to make it. True only when a request explicitly asks for a human, or the "
         "evidence is too incomplete/conflicting to route automatically. A clearly-worded, high-risk case "
         "still gets manual_routing_required=False if it routes cleanly, because a human reviewing it downstream (e.g. a "
         "surveillance analyst) is the normal queue, not a special flag. This is a judgment call made by one "
         "person (not independently reviewed) and at least one case in this set is a genuine toss-up under "
         "an equally defensible alternative rule - see jev.md's \"What a mistake actually looks like\" section."),
        ("Labels are unreviewed", "All ground-truth labels were written by one person for this benchmark. "
         "Treat them as a starting point, not an authority - if you disagree with one, you are probably not "
         "the first, and you're encouraged to relabel and re-run (see README.md)."),
        ("Confidence is not comparable across systems", "Jev's confidence is a returned probability. LLM "
         "confidence is a verbalised self-report (the model was asked to state one, and how well models do "
         "this varies). TF-IDF's is predict_proba from a plain classifier. Keyword rules have none. Treat "
         "cross-system confidence comparisons with real skepticism."),
        ("Repeats", "API-based systems (Jev, both LLMs) were called 3 times per case to get a sense of "
         "run-to-run variance (temperature, latency jitter). TF-IDF and keyword rules are deterministic given "
         "the same code and data, so they only have one row per case (repeat=0), repeated across the "
         "Compare sheet's repeat rows for alignment."),
        ("Where the raw API responses are", "This workbook has the parsed answers only. The full raw JSON "
         "response for every call is in results/raw.jsonl, one line per call, never overwritten between runs."),
    ]
    for r in rows:
        ws.append(r)
        ws.cell(row=ws.max_row, column=2).alignment = WRAP
    for r in (1,):
        ws.cell(row=r, column=1).font = Font(bold=True, size=14)
    for r in range(3, ws.max_row + 1):
        if ws.cell(row=r, column=1).value:
            ws.cell(row=r, column=1).font = Font(bold=True)


# --------------------------------------------------------------------------- Cases
def write_cases(wb, cases):
    ws = add_sheet(wb, "Cases")
    write_header(ws, ["case_id", "category", "request", "expected_agent", "acceptable_agents",
                      "manual_routing_required", "high_risk", "should_escalate"])
    for c in cases:
        ws.append([c["id"], c["category"], c["request"], c["expected_agent"],
                   ", ".join(c["acceptable_agents"]), c["manual_routing_required"], c["high_risk"], c["should_escalate"]])
        ws.cell(row=ws.max_row, column=3).alignment = WRAP
    autosize(ws, {1: 9, 2: 20, 3: 55, 4: 18, 5: 22, 6: 13, 7: 11, 8: 15})


# --------------------------------------------------------------------------- Responses (long)
RESP_HEADERS = ["case_id", "category", "request", "arm", "system", "model", "repeat",
                "pred_agent", "agent_confidence", "agent_correct",
                "pred_manual_routing_required", "manual_routing_required_confidence", "manual_routing_required_correct",
                "pred_high_risk", "high_risk_confidence", "high_risk_correct",
                "all_three_correct", "latency_s", "cost", "error"]


def score(case, d):
    if not d:
        return {"agent": False, "manual_routing_required": False, "high_risk": False}
    return {"agent": d["agent"][0] == case["expected_agent"],
            "manual_routing_required": d["manual_routing_required"][0] == case["manual_routing_required"],
            "high_risk": d["high_risk"][0] == case["high_risk"]}


def response_rows(cases_by_id, raw_rows, baseline_rows):
    out = []
    for r in raw_rows:
        c = cases_by_id[r["case_id"]]
        d = r.get("decisions")
        s = score(c, d)
        out.append([
            c["id"], c["category"], c["request"], r["arm"], r["system"], r["model"], r["repeat"],
            d["agent"][0] if d else None, d["agent"][1] if d else None, s["agent"],
            d["manual_routing_required"][0] if d else None, d["manual_routing_required"][1] if d else None, s["manual_routing_required"],
            d["high_risk"][0] if d else None, d["high_risk"][1] if d else None, s["high_risk"],
            all(s.values()) if d else False,
            r.get("latency_s"), (r.get("usage") or {}).get("cost"), r.get("error") or "",
        ])
    for r in baseline_rows:
        c = cases_by_id[r["case_id"]]
        d = r["decisions"]
        s = score(c, d)
        out.append([
            c["id"], c["category"], c["request"], r["arm"], r["system"], r["model"], r["repeat"],
            d["agent"][0], d["agent"][1], s["agent"],
            d["manual_routing_required"][0], d["manual_routing_required"][1], s["manual_routing_required"],
            d["high_risk"][0], d["high_risk"][1], s["high_risk"],
            all(s.values()), r.get("latency_s"), 0.0, "",
        ])
    return out


def write_responses(wb, rows):
    ws = add_sheet(wb, "Responses")
    write_header(ws, RESP_HEADERS)
    bool_cols = {RESP_HEADERS.index(c) + 1 for c in ("agent_correct", "manual_routing_required_correct",
                                                      "high_risk_correct", "all_three_correct")}
    for row in rows:
        ws.append(row)
        for col in bool_cols:
            fill_bool(ws.cell(row=ws.max_row, column=col), row[col - 1])
        ws.cell(row=ws.max_row, column=3).alignment = WRAP
    autosize(ws, {1: 9, 2: 20, 3: 50, 4: 26, 5: 10, 6: 26, 7: 8, "default": 13})


# --------------------------------------------------------------------------- Compare (wide)
def write_compare(wb, cases, arms_present, rows_by_arm_case_repeat, max_repeats):
    ws = add_sheet(wb, "Compare")
    headers = ["case_id", "category", "request", "repeat",
               "expected_agent", "expected_manual_routing_required", "expected_high_risk"]
    for arm in arms_present:
        label = arm.replace(":", " · ")
        headers += [f"{label}: agent", f"{label}: manual_routing_required", f"{label}: high_risk", f"{label}: all 3 ok"]
    write_header(ws, headers)

    bool_col_sets = []  # (all3_col,) per arm, plus per-field correctness handled via fill vs expected
    for c in cases:
        for repeat in range(max_repeats):
            row = [c["id"], c["category"], c["request"], repeat,
                   c["expected_agent"], c["manual_routing_required"], c["high_risk"]]
            per_arm_correct_cols = []
            for arm in arms_present:
                # TF-IDF/rules are deterministic: always show their one answer regardless of repeat.
                key = (arm, c["id"], repeat if arm not in ("tfidf_lr", "rules") else 0)
                d = rows_by_arm_case_repeat.get(key)
                if d:
                    s = score(c, d)
                    row += [d["agent"][0], d["manual_routing_required"][0], d["high_risk"][0], all(s.values())]
                    per_arm_correct_cols.append((len(row) - 4, s["agent"]))
                    per_arm_correct_cols.append((len(row) - 3, s["manual_routing_required"]))
                    per_arm_correct_cols.append((len(row) - 2, s["high_risk"]))
                    per_arm_correct_cols.append((len(row) - 1, all(s.values())))
                else:
                    row += [None, None, None, None]
            ws.append(row)
            r = ws.max_row
            ws.cell(row=r, column=3).alignment = WRAP
            for col_idx, ok in per_arm_correct_cols:
                fill_bool(ws.cell(row=r, column=col_idx), ok)
    widths = {1: 9, 2: 20, 3: 50, 4: 8, 5: 16, 6: 18, 7: 13}
    autosize(ws, widths)


# --------------------------------------------------------------------------- Summary
def write_summary(wb, summary_csv_path):
    ws = add_sheet(wb, "Summary")
    if not summary_csv_path.exists():
        ws.append(["Run analyze.py first to generate summary.csv"])
        return
    with summary_csv_path.open() as f:
        reader = csv.reader(f)
        header = next(reader)
        write_header(ws, header)
        for row in reader:
            ws.append(row)
    autosize(ws, {1: 20, 2: 16, "default": 15})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=Path(__file__).parent / "results")
    a = ap.parse_args()

    cases = spec.load_cases()
    cases_by_id = {c["id"]: c for c in cases}

    raw_path = a.dir / "raw.jsonl"
    raw_rows = analyze.load_raw(raw_path) if raw_path.exists() else []
    baseline_rows = baselines.run_rules(cases) + baselines.run_tfidf(cases)  # already carry "arm"

    meta_path = a.dir / "run_meta.json"
    run_meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    wb = Workbook()
    wb.remove(wb.active)  # drop the default empty sheet
    write_readme(wb, run_meta)
    write_cases(wb, cases)

    resp_rows = response_rows(cases_by_id, raw_rows, baseline_rows)
    write_responses(wb, resp_rows)

    # Index every response by (arm, case_id, repeat) for the wide Compare sheet.
    by_key = {}
    for r in raw_rows:
        if r.get("decisions"):
            by_key[(r["arm"], r["case_id"], r["repeat"])] = r["decisions"]
    for r in baseline_rows:
        by_key[(r["arm"], r["case_id"], r["repeat"])] = r["decisions"]
    arms_present = sorted({r["arm"] for r in raw_rows} | {r["arm"] for r in baseline_rows},
                          key=lambda arm: next((i for i, s in enumerate(SYSTEM_ORDER) if arm.startswith(s)), 99))
    max_repeats = max([r["repeat"] for r in raw_rows], default=0) + 1
    write_compare(wb, cases, arms_present, by_key, max_repeats)

    write_summary(wb, a.dir / "summary.csv")

    out = a.dir / "results.xlsx"
    wb.save(out)
    print(f"wrote {out} ({len(resp_rows)} response rows, {len(cases)} cases, {len(arms_present)} systems)")


if __name__ == "__main__":
    main()
