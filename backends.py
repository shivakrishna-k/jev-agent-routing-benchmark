"""API transports and response parsers.

Raw responses are stored untouched by benchmark.py; parsing happens at analysis time so a
parser fix never needs another paid run. `parse_*` return None on any malformed response
and analysis counts that as a failure.
"""
import json
import random
import time
import zlib

import requests

import spec

API = "https://openrouter.ai/api"
CHAT = f"{API}/v1/chat/completions"
DECISIONS = f"{API}/alpha/decisions"
MODELS = f"{API}/v1/models"


class ApiError(Exception):
    pass


class HttpTransport:
    def __init__(self, key, retries=3):
        self.h = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        self.retries = retries

    def get(self, url):
        r = requests.get(url, headers=self.h, timeout=30)
        r.raise_for_status()
        return r.json()

    def post(self, url, payload, timeout=90):
        """Returns (body, latency_s, attempts). Latency covers the successful attempt only,
        not any retries before it - retries measure infrastructure flakiness, not the model.

        Retry policy: 429 (rate limited) and 5xx (server error) are retried with exponential
        backoff, since those are usually transient. Any other 4xx (bad request, auth failure,
        unknown model) fails immediately - retrying a malformed request just wastes time and
        money, since the same request will fail the same way every time."""
        last = None
        for attempt in range(1, self.retries + 2):
            t0 = time.perf_counter()
            try:
                r = requests.post(url, headers=self.h, json=payload, timeout=timeout)
                dt = time.perf_counter() - t0
                if r.status_code == 429 or r.status_code >= 500:
                    last = f"HTTP {r.status_code}: {r.text[:200]}"
                elif r.status_code >= 400:
                    raise ApiError(f"HTTP {r.status_code}: {r.text[:300]}")
                else:
                    return r.json(), dt, attempt
            except (requests.Timeout, requests.ConnectionError) as e:
                last = f"{type(e).__name__}: {e}"
            time.sleep(min(2 ** attempt, 20))
        raise ApiError(f"gave up after {self.retries + 1} attempts: {last}")


# --------------------------------------------------------------------------- calls
# Each call_* function is one entry in CALLERS below (keyed by "arm system"), called once per
# (case, repeat) from benchmark.py. They all return the same shape: {"response": <raw API
# body, stored as-is>, "latency_s": <float>, "attempts": <retry count>} so benchmark.py's
# runner doesn't need to know which system it's calling.

def call_jev(tx, model, case):
    """One Jev call answers all 3 questions (agent/human_review/high_risk) against one shared
    state, in a single request - the "typed decision" interface the whole benchmark is about."""
    body, dt, n = tx.post(DECISIONS, spec.jev_payload(model, case["request"]), timeout=60)
    return {"response": body, "latency_s": dt, "attempts": n}


def _chat(tx, model, request, names, confidence):
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": spec.llm_system_prompt(names, confidence)},
            {"role": "user", "content": json.dumps({"request": request})},
        ],
        "response_format": {"type": "json_schema", "json_schema": {
            "name": "routing", "strict": True, "schema": spec.llm_schema(names, confidence)}},
        # Fail loudly rather than let a provider silently ignore the strict schema.
        "provider": {"require_parameters": True},
    }
    return tx.post(CHAT, payload, timeout=120)


def call_llm(tx, model, case):
    """The frontier-LLM baseline: one chat completion, strict JSON schema, answers all 3
    questions plus a self-reported confidence per question - the closest a plain LLM gets to
    Jev's interface, still one call per case."""
    body, dt, n = _chat(tx, model, case["request"], spec.QUESTION_NAMES, True)
    return {"response": body, "latency_s": dt, "attempts": n}


def call_llm_split(tx, model, case):
    """The "no shared decision layer" comparison arm: the SAME model as call_llm, but asked
    each of the 3 questions as an independent chat completion with no shared context between
    them, instead of one joint call. Tests what decomposing a decision costs when nothing
    holds the shared state - see jev.md's "Splitting one LLM into three separate calls" finding.

    One call per question, issued in parallel: latency = slowest of the 3 (not their sum,
    since they run concurrently), cost/attempts = the total across all 3."""
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(3) as ex:
        futs = {n: ex.submit(_chat, tx, model, case["request"], [n], True) for n in spec.QUESTION_NAMES}
        parts = {n: f.result() for n, f in futs.items()}
    return {"response": {"parts": {n: p[0] for n, p in parts.items()}},
            "latency_s": max(p[1] for p in parts.values()),
            "attempts": sum(p[2] for p in parts.values())}


CALLERS = {"jev": call_jev, "llm": call_llm, "llm_split": call_llm_split}


# --------------------------------------------------------------------------- parsers
# Every parse_* function takes the raw API body (exactly what was saved to raw.jsonl - nothing
# is transformed before saving) and returns either None (unparseable / malformed - counted as
# a failure everywhere downstream) or (decisions, usage) where `decisions` is always shaped
# {"agent": (choice: str, confidence: float|None), "manual_routing_required": (bool, confidence),
# "high_risk": (bool, confidence)} - one common shape all 5 systems get normalized into, so
# analyze.py and export_excel.py don't need to know which system produced a given row.

# manual_routing_required was originally called human_review (renamed because that name reads
# as "is this serious," a different question - see spec.py). Already-collected raw.jsonl rows
# were called with the old name on the wire (Jev's `answers` key, the LLM's JSON field name),
# so parsing falls back to it here rather than needing every past response re-collected.
LEGACY_KEYS = {"manual_routing_required": "human_review"}
CANON_KEYS = {v: k for k, v in LEGACY_KEYS.items()}  # old wire name -> today's name


def _legacy(name):
    """Today's name -> its pre-rename wire name, or itself if never renamed."""
    return LEGACY_KEYS.get(name, name)


def _canon(name):
    """A wire name (old or new) -> today's name, or itself if not a renamed field. Used for
    llm_split, where the OUTER dict key itself (not just an inner JSON field) is the old name
    in already-collected data - see call_llm_split, which keys "parts" by spec.QUESTION_NAMES
    as they were at call time."""
    return CANON_KEYS.get(name, name)

def _conf(x):
    """Clamp a self-reported LLM confidence into [0, 1]; None if it's missing or not a number
    (models occasionally return out-of-range values like 1.4 despite the schema asking for a
    probability - clamping rather than rejecting keeps the row usable)."""
    try:
        return min(1.0, max(0.0, float(x)))
    except (TypeError, ValueError):
        return None


def _max_prob(p):
    """Jev's `probabilities` for a choice question is a map like {"agent_a": 0.7, ...}; this
    is P(the chosen answer), i.e. the probability of whichever key had the highest value."""
    if isinstance(p, dict) and p:
        return max(float(v) for v in p.values())
    if isinstance(p, list) and p:
        return max(float(v.get("probability", v) if isinstance(v, dict) else v) for v in p)
    return None


def parse_jev(body):
    """Jev's Decisions API response shape: `answers.<question>` holds a `choice` (for `type:
    choice` questions) with a `probabilities` map, or a `noul` value (for `type: noul`
    yes/no questions) which IS P(true) directly - a noul of 0.95 means "95% confident this is
    true", not "95% confident in whatever the answer is". So a noul of 0.05 means the same
    confidence as 0.95, just for "false": the stored confidence is max(p, 1-p), and the
    boolean answer is simply p >= 0.5."""
    try:
        a = body["answers"]
        out = {}
        ag = a["agent"]
        out["agent"] = (ag["choice"], _max_prob(ag.get("probabilities")))
        for q in ("manual_routing_required", "high_risk"):
            node = a[q] if q in a else a[_legacy(q)]
            p = float(node["noul"])
            out[q] = (p >= 0.5, max(p, 1 - p))
        usage = body.get("usage") or {}
        return out, {"input_tokens": usage.get("input_tokens") or usage.get("prompt_tokens"),
                     "output_tokens": usage.get("output_tokens") or usage.get("completion_tokens"),
                     "cost": usage.get("cost")}
    except (KeyError, TypeError, ValueError, AttributeError):
        return None


def _parse_chat_content(body):
    """Chat completions return the model's JSON as a STRING inside .choices[0].message.content
    (that's the strict-JSON-schema contract), so it needs its own json.loads, separate from
    parsing the outer HTTP response body."""
    return json.loads(body["choices"][0]["message"]["content"]), body.get("usage") or {}


def parse_llm(body):
    """The single-call LLM arm: one chat completion whose JSON has all 3 answers plus a
    "<question>_confidence" field per question (see spec.llm_schema) - a verbalised, self-
    reported confidence, a fundamentally different mechanism from Jev's returned probability
    (see jev.md's calibration section for why that comparison needs care)."""
    try:
        c, u = _parse_chat_content(body)
        out = {}
        for n in spec.QUESTION_NAMES:
            val = c[n] if n in c else c[_legacy(n)]
            conf = c.get(f"{n}_confidence", c.get(f"{_legacy(n)}_confidence"))
            out[n] = (val, _conf(conf))
        if out["agent"][0] not in spec.AGENTS:  # schema allows any string; guard against a stray value anyway
            return None
        return out, {"input_tokens": u.get("prompt_tokens"), "output_tokens": u.get("completion_tokens"),
                     "cost": u.get("cost")}
    except (KeyError, IndexError, TypeError, ValueError):
        return None


def parse_llm_split(body):
    """The 3-separate-calls arm: body["parts"] holds one chat-completion response per question
    (see call_llm_split), each parsed the same way as parse_llm's single call, then summed for
    usage/cost. Requires ALL 3 parts to parse successfully and be present - a partial result
    (e.g. one of the three calls failed) is treated as a total failure, same as the other arms,
    rather than silently scoring on 2 of 3 questions."""
    try:
        out, tot = {}, {"input_tokens": 0, "output_tokens": 0, "cost": 0.0}
        for n, b in body["parts"].items():
            canon = _canon(n)  # today's name for this part, whichever wire name was actually used
            c, u = _parse_chat_content(b)
            val = c[n] if n in c else c[canon]
            conf = c.get(f"{n}_confidence", c.get(f"{canon}_confidence"))
            out[canon] = (val, _conf(conf))
            tot["input_tokens"] += u.get("prompt_tokens") or 0
            tot["output_tokens"] += u.get("completion_tokens") or 0
            tot["cost"] += u.get("cost") or 0.0
        if out["agent"][0] not in spec.AGENTS or set(out) != set(spec.QUESTION_NAMES):
            return None
        return out, tot
    except (KeyError, IndexError, TypeError, ValueError):
        return None


PARSERS = {"jev": parse_jev, "llm": parse_llm, "llm_split": parse_llm_split}


# --------------------------------------------------------------------------- simulator
class MockTransport:
    """Offline simulator used by --dry-run. Produces responses in the assumed API shapes
    so the runner/parsers/analysis/plots can be exercised without spending credits.
    Numbers it generates are FAKE and carry no information about Jev or any LLM."""

    ACC = {"clear": .99, "adversarial": .88, "ambiguous": .70, "missing_evidence": .72, "conflicting_evidence": .68}

    def __init__(self, cases):
        self.by_req = {c["request"]: c for c in cases}

    def _rng(self, *k):
        return random.Random(zlib.crc32("|".join(map(str, k)).encode()) + int(time.time() * 1e6) % 1000)

    def _find(self, payload):
        if "state" in payload:
            return self.by_req[payload["state"]["request"]]
        return self.by_req[json.loads(payload["messages"][1]["content"])["request"]]

    def get(self, url):
        return {"data": [{"id": m} for m in ("mock/llm-a", "mock/llm-b")]}

    def post(self, url, payload, timeout=0):
        case = self._find(payload)
        rng = self._rng(url, case["id"], payload.get("model"), json.dumps(sorted(payload.get("questions", {}))))
        acc = self.ACC[case["category"]]
        agents = list(spec.AGENTS)

        def pick(truth_ok, truth, options):
            return truth if truth_ok else rng.choice([o for o in options if o != truth])

        if url == DECISIONS:
            dt = rng.uniform(0.25, 0.7)
            ok = rng.random() < acc
            conf = rng.uniform(.86, .995) if ok else rng.uniform(.45, .8)
            choice = pick(ok, case["expected_agent"], agents)
            probs = {a: (conf if a == choice else (1 - conf) / 3) for a in agents}
            ans = {"agent": {"choice": choice, "probabilities": probs}}
            for q in ("manual_routing_required", "high_risk"):
                ok = rng.random() < acc
                c = rng.uniform(.85, .99) if ok else rng.uniform(.5, .78)
                truth = case[q] if ok else (not case[q])
                ans[q] = {"noul": c if truth else 1 - c}
            return {"answers": ans, "usage": {"input_tokens": 260, "output_tokens": 0, "cost": 0.00004}}, dt, 1
        # chat completion (single or split)
        names = list(json.loads(json.dumps(payload["response_format"]["json_schema"]["schema"]["properties"])))
        names = [n for n in names if not n.endswith("_confidence")]
        dt = rng.uniform(1.2, 4.5)
        out = {}
        for n in names:
            ok = rng.random() < (acc - .02)
            truth = case["expected_agent"] if n == "agent" else case[n]
            val = pick(ok, truth, agents) if n == "agent" else (truth if ok else (not truth))
            out[n] = val
            out[f"{n}_confidence"] = rng.choice([.9, .95, .95, .97, .98, .99])  # over-confident
        body = {"choices": [{"message": {"content": json.dumps(out)}}],
                "usage": {"prompt_tokens": 420, "completion_tokens": 40 * len(names), "cost": 0.0011 * len(names)}}
        return body, dt, 1
