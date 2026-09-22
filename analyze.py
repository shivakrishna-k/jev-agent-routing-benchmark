"""Computes all metrics from raw.jsonl (+ offline baselines).

    python analyze.py [--dir results]     ->  metrics.json, summary.csv, summary.md

Conventions
  * A failed call (API error or unparseable response) counts as INCORRECT for accuracy and is
    excluded from latency, cost and calibration. The failure rate is reported per arm.
  * "decision" = one answered question; a case has 3 decisions. Cost is per 1,000 decisions.
  * Confidence: Jev = returned probability; LLM = verbalised 0-1 self-report; TF-IDF = predict_proba;
    rules = none. Case confidence = min over the three decisions.
  * With --repeats > 1 rows are pooled, so the Wilson intervals are optimistic (repeats of the
    same case are not independent).
"""
import argparse
import warnings
import json
import math
from collections import defaultdict
from pathlib import Path

import backends
import baselines
import metrics as M
import spec

warnings.filterwarnings("ignore", category=UserWarning)
HI, LO = 0.90, 0.60  # default routing policy thresholds
# Color follows the entity, never its position: a system keeps its color whichever arms are present.
SYSTEM_COLOR = {"jev": "#2a78d6", "llm_split": "#e87ba4", "tfidf_lr": "#4a3aa7", "rules": "#898781"}
LLM_COLORS = ["#eb6834", "#1baf7a", "#eda100", "#008300"]


def load_raw(path):
    """Reads raw.jsonl and parses each row's stored API response into a `decisions` dict
    (or None if the call errored or the response didn't match the expected shape). Parsing
    happens here, at analysis time, rather than when the response was first saved, so a
    parser bug can be fixed and rerun without spending API credits again."""
    rows = []
    for line in Path(path).open():
        r = json.loads(line)
        parsed = None if r.get("error") else backends.PARSERS[r["system"]](r.get("response"))
        r["decisions"], r["usage"] = (parsed if parsed else (None, {}))
        rows.append(r)
    return rows


def arm_meta(arm, system, model, llm_index):
    """Display label and chart color for one arm. Colors are assigned by identity (which
    LLM this is) rather than by the order arms happen to appear in raw.jsonl - that order is
    nondeterministic, since rows are written in whichever order the concurrent API calls
    finish (see benchmark.py's ThreadPoolExecutor), not the order they were submitted."""
    short = model.split("/")[-1]
    label = {"jev": "Jev", "llm": short, "llm_split": f"{short} (3 calls)",
             "rules": "Keyword rules", "tfidf_lr": "TF-IDF + LR"}[system]
    color = LLM_COLORS[llm_index % len(LLM_COLORS)] if system == "llm" else SYSTEM_COLOR[system]
    return {"label": label, "system": system, "model": model, "color": color}


def evaluate(rows, cases_by_id):
    """Computes every metric for one arm (one system) from its rows of raw results.

    `rows` is every (case, repeat) attempt for this one arm; a "decision" is one of the three
    questions (agent / manual_routing_required / high_risk) answered for one case, so each row
    holds up to 3 decisions. Failed calls (r["decisions"] is None) count as wrong on every
    question and are excluded from latency, cost and calibration, since there's no response to
    measure."""
    n_rows = len(rows)
    ok = [r for r in rows if r["decisions"]]
    out = {"n_rows": n_rows, "n_ok": len(ok), "failure_rate": 1 - len(ok) / n_rows if n_rows else float("nan")}

    # Score every row against its case's ground truth before aggregating anything.
    per_row = []  # (row, case, correct{question: bool}, all_three_correct, agent_correct_leniently)
    for r in rows:
        c = cases_by_id[r["case_id"]]
        d = r["decisions"]
        if d:
            corr = {"agent": d["agent"][0] == c["expected_agent"],
                    "manual_routing_required": d["manual_routing_required"][0] == c["manual_routing_required"],
                    "high_risk": d["high_risk"][0] == c["high_risk"]}
            # "Lenient" agent scoring credits any agent a reasonable reviewer would also accept
            # (see acceptable_agents in data/build_dataset.py), used only for the agent_lenient
            # stat - the headline "all three correct" figure always uses the strict single label.
            lenient = d["agent"][0] in c["acceptable_agents"]
        else:
            corr, lenient = {q: False for q in spec.QUESTION_NAMES}, False
        per_row.append((r, c, corr, all(corr.values()), lenient))

    def rate(vals):
        """A proportion plus its 95% Wilson confidence interval, given a list of booleans."""
        k, n = sum(vals), len(vals)
        lo, hi = M.wilson(k, n)
        return {"rate": k / n if n else float("nan"), "lo": lo, "hi": hi, "n": n}

    out["accuracy"] = {q: rate([p[2][q] for p in per_row]) for q in spec.QUESTION_NAMES}
    out["accuracy"]["all_three"] = rate([p[3] for p in per_row])
    out["accuracy"]["agent_lenient"] = rate([p[4] for p in per_row])

    # The headline number. `rate()`'s Wilson interval above pools every (case, repeat) row as
    # if they were 300 independent draws, but repeats of the same case are correlated (a hard
    # case tends to be hard on every repeat), so that interval is optimistic - see metrics.py's
    # cluster_bootstrap_ci docstring. Collapse each case to its own mean correctness first
    # (one summary value per case, across that case's repeats), then bootstrap over CASES:
    # this is the number actually reported as a confidence interval in jev.md, and is also
    # what main() uses below for the paired system-vs-Jev comparisons.
    by_case = defaultdict(list)
    for r, c, corr, all_ok, _ in per_row:
        by_case[c["id"]].append(all_ok)
    case_values = {cid: sum(v) / len(v) for cid, v in by_case.items()}
    case_lo, case_hi = M.cluster_bootstrap_ci(list(case_values.values()))
    out["accuracy"]["all_three"]["case_bootstrap"] = {"lo": case_lo, "hi": case_hi, "method": "cluster bootstrap by case, 10000 resamples"}
    out["accuracy"]["all_three"]["case_values"] = case_values

    # Same "all three correct" rate, broken out by how hard the case was (its category).
    cats = defaultdict(list)
    for p in per_row:
        cats[p[1]["category"]].append(p[3])
    out["by_category"] = {k: rate(v) for k, v in cats.items()}

    # manual_routing_required confusion matrix: "positive" means the case's ground truth says
    # a person was needed to make the routing decision. A failed call is counted as a negative
    # prediction (fn if truth was positive), since a system that couldn't answer certainly
    # didn't flag it for a human either.
    tp = fp = fn = tn = 0
    for r, c, corr, _, _ in per_row:
        pred = r["decisions"]["manual_routing_required"][0] if r["decisions"] else None
        truth = c["manual_routing_required"]
        if pred is None:
            fn += truth
            continue
        tp += pred and truth; fp += pred and not truth
        fn += (not pred) and truth; tn += (not pred) and not truth
    out["manual_routing_confusion"] = {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
                                       "missed_escalation_rate": fn / (tp + fn) if tp + fn else float("nan"),
                                       "unneeded_escalation_rate": fp / (fp + tn) if fp + tn else float("nan")}

    # Latency/cost/tokens only make sense for calls that actually returned something.
    lat = [r["latency_s"] for r in ok]
    out["latency_s"] = {"p50": M.percentile(lat, 50), "p95": M.percentile(lat, 95), "n": len(lat)}
    costs = [r["usage"].get("cost") for r in ok if r["usage"].get("cost") is not None]
    # Cost is normalized per DECISION (3 per case/row), not per case, so systems that answer a
    # different number of questions per call (e.g. the 3-separate-calls arm) are comparable.
    out["cost_per_1k_decisions"] = (sum(costs) / (len(costs) * 3) * 1000) if costs else None
    toks = [(r["usage"].get("input_tokens") or 0) + (r["usage"].get("output_tokens") or 0) for r in ok]
    out["tokens_per_case"] = sum(toks) / len(toks) if toks else None

    # Calibration (does confidence mean anything?) and the confidence-routing simulation both
    # need every decision to carry a confidence value. Keyword rules have none, so both blocks
    # are skipped (left as None) for that arm - see visuals.py, which renders "n/a" for those.
    confs, corrs, case_conf, case_corr, case_esc = [], [], [], [], []
    has_conf = bool(ok) and all(v[1] is not None for r in ok for v in r["decisions"].values())
    if has_conf:
        for r, c, corr, all_ok, _ in per_row:
            if not r["decisions"]:
                continue
            for q in spec.QUESTION_NAMES:
                confs.append(r["decisions"][q][1]); corrs.append(corr[q])
            # A case's overall confidence is the MIN across its three decisions: a routing
            # policy that auto-approves a case needs every decision about it to be trustworthy,
            # not just the average, so the weakest link sets the case's confidence.
            case_conf.append(min(v[1] for v in r["decisions"].values()))
            case_corr.append(all_ok); case_esc.append(c["should_escalate"])
        out["calibration"] = {
            "n": len(confs), "ece": M.ece(confs, corrs), "brier": M.brier(confs, corrs),
            "auroc_error_detection": M.auroc_error_detection(confs, corrs),
            "mean_confidence": sum(confs) / len(confs), "accuracy": sum(corrs) / len(corrs),
            "bins": M.reliability_bins(confs, corrs)}
        out["routing"] = {"curve": M.risk_coverage(case_conf, case_corr)}

        # Simulate the 3-tier routing policy described in the write-up: auto-approve above HI,
        # send to a (hypothetical) deeper-reasoning step between LO and HI, else to a human.
        auto = [x >= HI for x in case_conf]
        deep = [LO <= x < HI for x in case_conf]
        n = len(case_conf)
        wrong = [not c for c in case_corr]
        na = sum(auto)
        MIN_AUTO_N = 10  # below this, "accuracy when auto-approved" is a coin flip, not a measurement
        out["routing"]["policy"] = {
            "hi": HI, "lo": LO, "auto_n": na,
            "auto_share": na / n, "deeper_reasoning_share": sum(deep) / n, "human_share": (n - na - sum(deep)) / n,
            # nan (not 0) below MIN_AUTO_N: with too few auto-approved cases this is noise, not
            # a measurement, and displaying a precise-looking percentage would be misleading.
            "auto_accuracy": (sum(c for c, a in zip(case_corr, auto) if a) / na) if na >= MIN_AUTO_N else float("nan"),
            # Of the cases this system got wrong, what share did the policy catch (i.e. did NOT
            # auto-approve)? Answers "if I trust the low-confidence flag, how many bad calls do I avoid?"
            "errors_caught": (sum(w and not a for w, a in zip(wrong, auto)) / sum(wrong)) if any(wrong) else float("nan"),
            # Precision/recall of "not auto-approved" against the dataset's own should_escalate
            # label - a different question from errors_caught: does low confidence line up with
            # cases that were *designed* to be hard, not just cases this particular system muffed?
            "escalation_precision": (sum(e for e, a in zip(case_esc, auto) if not a) / (n - na)) if n - na else float("nan"),
            "escalation_recall": (sum(not a for a, e in zip(auto, case_esc) if e) / sum(case_esc)) if any(case_esc) else float("nan")}
    else:
        out["calibration"] = None
        out["routing"] = None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=Path(__file__).parent / "results")
    a = ap.parse_args()
    cases = spec.load_cases()
    by_id = {c["id"]: c for c in cases}
    # API-based systems come from the saved run; the two offline baselines are computed fresh
    # every time analyze.py runs (they're fast and deterministic, so there's nothing to cache).
    raw = load_raw(a.dir / "raw.jsonl")
    raw += baselines.run_rules(cases) + baselines.run_tfidf(cases)

    # Group every row by which arm (system) produced it, then put the arms in a fixed,
    # readable order for the charts and tables (jev, then LLMs, then the baselines) - the
    # order rows appear in raw.jsonl is not meaningful (see the note on arm_meta above).
    arms = defaultdict(list)
    for r in raw:
        arms[r["arm"]].append(r)
    meta_p = a.dir / "run_meta.json"
    run_meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
    order = sorted(arms, key=lambda k: (["jev", "llm", "llm_split", "tfidf_lr", "rules"].index(arms[k][0]["system"]), k))
    result = {"meta": run_meta, "dry_run": run_meta.get("dry_run", False), "arms": {}, "n_cases": len(cases),
              "category_counts": {k: sum(c["category"] == k for c in cases) for k in dict.fromkeys(c["category"] for c in cases)}}
    # llm_index picks which color an LLM arm gets (see arm_meta) - by position among LLM arms
    # specifically, so adding/removing baselines never reshuffles which LLM is which color.
    llm_arms = [k for k in order if arms[k][0]["system"] == "llm"]
    for arm in order:
        r0 = arms[arm][0]
        idx = llm_arms.index(arm) if arm in llm_arms else 0
        result["arms"][arm] = {**arm_meta(arm, r0["system"], r0["model"], idx), **evaluate(arms[arm], by_id)}

    # Paired comparisons, Jev vs. each single-call LLM: are they ACTUALLY different, given they
    # were run on the exact same 100 cases? A paired bootstrap (metrics.paired_bootstrap_diff)
    # answers this properly, unlike eyeballing whether two separate confidence intervals
    # overlap (overlapping CIs don't rule out a real difference, and non-overlapping ones don't
    # by themselves confirm one either). Limited to "llm" arms (not llm_split/baselines) since
    # that's the comparison the write-up's headline accuracy claim actually makes.
    jev_arm = next((k for k in order if arms[k][0]["system"] == "jev"), None)
    result["pairwise_vs_jev"] = {}
    if jev_arm:
        jev_cv = result["arms"][jev_arm]["accuracy"]["all_three"]["case_values"]
        jev_vals = [jev_cv[c["id"]] for c in cases]
        for arm in llm_arms:
            other_cv = result["arms"][arm]["accuracy"]["all_three"]["case_values"]
            other_vals = [other_cv[c["id"]] for c in cases]
            point, lo, hi, p = M.paired_bootstrap_diff(other_vals, jev_vals)
            result["pairwise_vs_jev"][arm] = {"label": result["arms"][arm]["label"], "diff": point,
                                              "lo": lo, "hi": hi, "p": p, "n_boot": 10000,
                                              "note": "diff = this system's all-3-correct rate minus Jev's, paired bootstrap by case"}

    # NaN isn't valid JSON; write it as null rather than crashing json.dumps or emitting "NaN".
    (a.dir / "metrics.json").write_text(json.dumps(result, indent=1, default=lambda x: None if isinstance(x, float) and math.isnan(x) else x))
    write_summary(a.dir, result)
    print(f"wrote {a.dir}/metrics.json, summary.csv, summary.md")


def _f(x, pct=False, nd=3):
    """Format a metric for the human-readable summary: nan/None -> "n/a", never a bare 'NaN'."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{x * 100:.0f}%" if pct else f"{x:.{nd}f}"


def write_summary(d, res):
    """Writes the plain-text summary.md (for a quick read) and summary.csv (for a spreadsheet)
    from the same `metrics.json`-shaped `res` dict evaluate()/main() built. This is the
    lightweight, per-system-aggregate view; results.xlsx (export_excel.py) has the full,
    per-case detail behind these numbers."""
    lines = ["| System | All-3 accuracy (95% CI, cluster bootstrap by case) | Agent | Manual routing | High-risk | ECE | Error-detect AUROC | p50 / p95 s | $ / 1K decisions | Fail |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    csv = ["arm,label,n_rows,failure_rate,acc_all,acc_all_lo_naive_pooled,acc_all_hi_naive_pooled,acc_all_lo_cluster_bootstrap,acc_all_hi_cluster_bootstrap,acc_agent,acc_manual_routing_required,acc_high_risk,ece,brier,auroc,p50_s,p95_s,cost_per_1k_decisions,missed_escalation_rate"]
    for arm, m in res["arms"].items():
        a, c = m["accuracy"], m["calibration"]
        cost = m["cost_per_1k_decisions"]
        cb = a["all_three"]["case_bootstrap"]
        lines.append(f"| {m['label']} | {_f(a['all_three']['rate'], True)} ({_f(cb['lo'], True)}-{_f(cb['hi'], True)}) "
                     f"| {_f(a['agent']['rate'], True)} | {_f(a['manual_routing_required']['rate'], True)} | {_f(a['high_risk']['rate'], True)} "
                     f"| {_f(c['ece']) if c else 'n/a'} | {_f(c['auroc_error_detection'], nd=2) if c else 'n/a'} "
                     f"| {_f(m['latency_s']['p50'], nd=2)} / {_f(m['latency_s']['p95'], nd=2)} | {'$' + _f(cost, nd=2) if cost is not None else 'n/a'} | {_f(m['failure_rate'], True)} |")
        csv.append(",".join(str(x) for x in [
            arm, m["label"], m["n_rows"], m["failure_rate"], a["all_three"]["rate"], a["all_three"]["lo"], a["all_three"]["hi"],
            cb["lo"], cb["hi"], a["agent"]["rate"], a["manual_routing_required"]["rate"], a["high_risk"]["rate"],
            c["ece"] if c else "", c["brier"] if c else "", c["auroc_error_detection"] if c else "",
            m["latency_s"]["p50"], m["latency_s"]["p95"], cost if cost is not None else "",
            m["manual_routing_confusion"]["missed_escalation_rate"]]))
    banner = "> **DRY RUN: simulated numbers, not real results.**\n\n" if res["dry_run"] else ""
    note = ("_The interval above resamples cases (not the pooled case-repeat rows), which properly accounts for the "
            "3 repeats per case being correlated rather than independent; it is therefore wider than a naive pooled "
            "interval would be. Both are in summary.csv if you want to compare._")
    cats = list(res["category_counts"])
    lines += ["", "**All-3 accuracy by case category**", "",
              "| System | " + " | ".join(f"{k} (n={res['category_counts'][k]})" for k in cats) + " |",
              "|---|" + "---|" * len(cats)]
    for m in res["arms"].values():
        lines.append(f"| {m['label']} | " + " | ".join(_f(m["by_category"][k]["rate"], True) for k in cats) + " |")
    pol = ["", f"**Confidence routing** (auto >= {HI}, deeper reasoning {LO}-{HI}, human < {LO}; case confidence = min over 3 decisions)", "",
           "| System | Auto share | Auto accuracy | Errors caught | Escalation precision | Escalation recall |", "|---|---|---|---|---|---|"]
    for m in res["arms"].values():
        p = (m["routing"] or {}).get("policy")
        if p:
            pol.append(f"| {m['label']} | {_f(p['auto_share'], True)} | {_f(p['auto_accuracy'], True)} | {_f(p['errors_caught'], True)} | {_f(p['escalation_precision'], True)} | {_f(p['escalation_recall'], True)} |")
    hr = ["", "**Manual-routing errors** (positive = a person was needed to make the routing decision)", "", "| System | Missed escalations | Unneeded escalations |", "|---|---|---|"]
    for m in res["arms"].values():
        c = m["manual_routing_confusion"]
        hr.append(f"| {m['label']} | {c['fn']} ({_f(c['missed_escalation_rate'], True)}) | {c['fp']} ({_f(c['unneeded_escalation_rate'], True)}) |")
    pw = ["", "**System vs. Jev, all-3-correct rate** (paired bootstrap by case: same cases resampled for both systems, so the difference accounts for cases both systems find hard/easy together)", "",
          "| System | Difference vs. Jev | 95% CI | Two-sided p |", "|---|---|---|---|"]
    for arm, r in res.get("pairwise_vs_jev", {}).items():
        pw.append(f"| {r['label']} | {r['diff']*100:+.1f} pp | ({r['lo']*100:+.1f}, {r['hi']*100:+.1f}) pp | {r['p']:.3f} |")
    (d / "summary.md").write_text(banner + "\n".join(lines) + "\n\n" + note + "\n" + "\n".join(pol + hr + pw) + "\n")
    (d / "summary.csv").write_text("\n".join(csv) + "\n")


if __name__ == "__main__":
    main()
