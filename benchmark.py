"""Runs Jev and LLM baselines on the routing cases and appends raw responses to raw.jsonl.

    python benchmark.py --smoke              # 1 case per system, prints raw responses, no file written
    python benchmark.py --limit 10           # quick trial
    python benchmark.py --repeats 3 --split  # full run, 3 repeats, plus the 3-calls-per-case arm
    python benchmark.py --dry-run            # offline simulator -> results/dryrun/ (FAKE numbers)

Runs are resumable: re-running skips (case, arm, repeat) already present in raw.jsonl.
Then: python analyze.py && python visuals.py
"""
import argparse
import datetime as dt
import hashlib
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv

import backends
import spec

ROOT = Path(__file__).parent
load_dotenv(ROOT / ".env")


def arms_from_env(split):
    """Which (system, model) pairs to run, read from .env. Always Jev plus whichever LLM_MODEL_*
    are set (1 or 2); --split adds a 3rd arm reusing LLM_MODEL_A as three separate calls
    instead of one joint call (see backends.call_llm_split)."""
    jev = os.getenv("JEV_MODEL", "typesafe/jev-1.13")
    llms = [m for m in (os.getenv("LLM_MODEL_A"), os.getenv("LLM_MODEL_B")) if m]
    arms = [("jev", jev)] + [("llm", m) for m in llms]
    if split and llms:
        arms.append(("llm_split", llms[0]))
    return arms


def key_looks_valid(k):
    """A cheap sanity check before making any API call: OpenRouter keys start "sk-or-". Catches
    the easy mistake of pasting an OpenAI/Anthropic key into .env by accident."""
    return k.startswith("sk-or-") and 40 < len(k) < 120


def preflight(tx, arms, cases, dry):
    """Fail fast on bad key / unknown model / unparseable Jev shape before spending money."""
    if not dry:
        key = os.environ.get("OPENROUTER_API_KEY", "")
        if not key_looks_valid(key):
            sys.exit("OPENROUTER_API_KEY in .env does not look like an OpenRouter key "
                     "(expected 'sk-or-...'). Fix .env and retry.")
        try:
            listed = {m["id"] for m in tx.get(backends.MODELS)["data"]}
        except Exception as e:
            sys.exit(f"Could not list models (bad key?): {e}")
        for system, model in arms:
            # Jev is served via the Decisions API and is not in /models, so only LLMs are checked here.
            if system != "jev" and model not in listed:
                close = sorted(m for m in listed if m.split("/")[0] == model.split("/")[0])[:8]
                sys.exit(f"Model '{model}' not found on OpenRouter. Same-provider IDs: {close}")
    for system, model in arms:
        try:
            res = backends.CALLERS[system](tx, model, cases[0])
        except Exception as e:
            sys.exit(f"Preflight call failed for {system}:{model}: {e}")
        if backends.PARSERS[system](res["response"]) is None:
            sys.exit(f"Preflight: could not parse {system}:{model} response. Raw:\n"
                     f"{json.dumps(res['response'])[:1500]}\nAdjust the parser in backends.py.")
    print("preflight ok")


def done_keys(path):
    """(case_id, arm, repeat) triples already successfully recorded in raw.jsonl, so a rerun
    (after a crash, a rate limit, or just wanting more repeats) only fills in what's missing
    instead of re-paying for and overwriting everything. A row with an error is NOT counted as
    done, so a failed call gets retried on the next run rather than staying failed forever."""
    keys = set()
    if path.exists():
        for line in path.open():
            r = json.loads(line)
            if not r.get("error"):
                keys.add((r["case_id"], r["arm"], r["repeat"]))
    return keys


def run_one(tx, system, model, case, repeat, run_id):
    """Makes one API call and returns a raw.jsonl row - always, even on failure (with an
    "error" field instead of a "response"), so a crash mid-run never silently drops a case."""
    rec = {"run_id": run_id, "ts": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
           "case_id": case["id"], "arm": f"{system}:{model}", "system": system, "model": model,
           "repeat": repeat}
    try:
        rec.update(backends.CALLERS[system](tx, model, case))
    except Exception as e:  # recorded, counted as a failure in analysis
        rec["error"] = str(e)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--split", action="store_true", help="also run the 3-separate-calls LLM arm")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()

    cases = spec.load_cases()
    if a.limit:
        cases = cases[:a.limit]
    arms = arms_from_env(a.split)
    out_dir = a.out or ROOT / "results" / ("dryrun" if a.dry_run else "")
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = out_dir / "raw.jsonl"

    tx = backends.MockTransport(spec.load_cases()) if a.dry_run else backends.HttpTransport(os.environ.get("OPENROUTER_API_KEY", ""))

    if a.smoke:
        for system, model in arms:
            r = run_one(tx, system, model, cases[0], 0, "smoke")
            print(f"\n=== {system}:{model} ({r.get('latency_s', 0):.2f}s)\n{json.dumps(r.get('response', r), indent=1)[:2500]}")
            print("parsed:", backends.PARSERS[system](r["response"]) if "response" in r else None)
        return

    preflight(tx, arms, cases, a.dry_run)

    run_id = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    meta = {"run_id": run_id, "date": run_id[:8], "arms": [f"{s}:{m}" for s, m in arms],
            "n_cases": len(cases), "repeats": a.repeats, "dry_run": a.dry_run,
            "dataset_sha256": hashlib.sha256((ROOT / "data" / "cases.json").read_bytes()).hexdigest()[:16],
            "prompt_sha256": hashlib.sha256(spec.llm_system_prompt().encode()).hexdigest()[:16]}
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=1) + "\n")

    done = done_keys(raw)
    jobs = [(s, m, c, r) for r in range(a.repeats) for c in cases for s, m in arms
            if (c["id"], f"{s}:{m}", r) not in done]
    print(f"{len(jobs)} calls to run ({len(done)} already done) -> {raw}")

    # Jobs run concurrently (a.workers at a time) and are written to raw.jsonl as they finish,
    # not in submission order - so the file's row order reflects which call happened to return
    # first, not case/arm/repeat order. That's fine for analysis (nothing depends on file
    # order) but worth knowing if you're looking at the raw file directly. Opening in append
    # mode + a lock around each write means a crash partway through never corrupts already-
    # written rows, and a rerun's done_keys() check picks up exactly where it left off.
    lock, n_err = threading.Lock(), 0
    with raw.open("a") as f, ThreadPoolExecutor(a.workers) as ex:
        futs = [ex.submit(run_one, tx, s, m, c, r, run_id) for s, m, c, r in jobs]
        for i, fut in enumerate(as_completed(futs), 1):
            rec = fut.result()
            with lock:
                f.write(json.dumps(rec) + "\n")
                f.flush()  # so `tail -f raw.jsonl` and a Ctrl-C mid-run both see real progress
            n_err += bool(rec.get("error"))
            if i % 25 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} done, {n_err} errors")
    print("done. Next: python analyze.py" + (" --dir results/dryrun" if a.dry_run else ""))


if __name__ == "__main__":
    main()
