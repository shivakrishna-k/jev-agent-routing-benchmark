"""Single source of truth for the decision questions.

Both Jev and the LLM baselines are generated from these definitions, so neither system
gets hints (or definitions) the other does not.
"""
import json
from pathlib import Path

DATA = Path(__file__).parent / "data" / "cases.json"

AGENTS = {
    "trade_surveillance": "Investigates unusual or potentially suspicious trading activity and surveillance alerts.",
    "market_risk": "Analyzes market exposure, risk limits, sensitivities and stress.",
    "compliance": "Handles regulatory policy, governance, controls and compliance interpretation.",
    "human": "Requires direct human handling because a person is explicitly needed or no specialist agent can own the request.",
}

QUESTIONS = {
    "agent": {
        "type": "choice",
        "instructions": "Which agent should handle this request first?",
        "criteria": AGENTS,
    },
    # Renamed from "human_review" (see data/build_dataset.py's docstring): the old name read as
    # "is this serious," a broader question than the narrower one actually being asked here.
    "manual_routing_required": {
        "type": "noul",
        "instructions": "Does the routing decision itself need a person to make it, rather than being routed automatically to an agent?",
        "criteria": {
            "true": "A person must make or approve the routing decision: it is explicitly requested, needs sign-off, or the evidence is too incomplete or conflicting to route automatically.",
            "false": "The request can be routed automatically, without a person deciding the routing itself.",
        },
    },
    "high_risk": {
        "type": "noul",
        "instructions": "Does this request involve a high-risk activity or consequence?",
        "criteria": {
            "true": "Material regulatory, conduct, market or other high-risk consequence.",
            "false": "Routine or low-risk activity.",
        },
    },
}
QUESTION_NAMES = list(QUESTIONS)


def load_cases(path=DATA):
    """Loads the 100 benchmark cases (see data/build_dataset.py for how they were built)."""
    return json.loads(Path(path).read_text())


def jev_payload(model, request):
    """Jev's Decisions API request body: the same QUESTIONS dict used to build the LLM prompt
    and schema below, so Jev is asked exactly the same three questions in its own native
    format - "choice" and "noul" types, not a description of them in English."""
    return {"model": model, "state": {"request": request}, "questions": QUESTIONS}


def _describe(name, q):
    """Renders one question's instructions + criteria as plain English, for the LLM prompt."""
    lines = [f"Question `{name}`: {q['instructions']}"]
    lines += [f"  - {k}: {v}" for k, v in q["criteria"].items()]
    return "\n".join(lines)


def llm_system_prompt(names=QUESTION_NAMES, confidence=True):
    """Builds the LLM baseline's system prompt FROM the same QUESTIONS dict Jev receives
    natively - the whole point is that no system gets a definition or hint the other lacks.
    `names` lets call_llm_split ask about just one question at a time; `confidence` controls
    whether the prompt asks for a verbalised self-reported confidence per answer."""
    parts = [
        "You are an enterprise AI routing decision engine.",
        "Answer each question below for the request, using only the definitions given.",
        "Return only JSON matching the schema.",
        "",
        *[_describe(n, QUESTIONS[n]) for n in names],
    ]
    if confidence:
        parts += ["", "For every answer also give <name>_confidence: your probability (0 to 1) that the answer is correct."]
    return "\n".join(parts)


def llm_schema(names=QUESTION_NAMES, confidence=True):
    """The strict JSON schema paired with llm_system_prompt (see backends._chat). `agent` is
    a closed enum of the 4 agent names; the other two questions are plain booleans, matching
    Jev's choice/noul types respectively. additionalProperties: False plus every field in
    `required` is what makes this "strict" - the API rejects a response missing a field or
    adding an extra one, rather than silently accepting a malformed answer."""
    props, req = {}, []
    for n in names:
        if n == "agent":
            props[n] = {"type": "string", "enum": list(AGENTS)}
        else:
            props[n] = {"type": "boolean"}
        req.append(n)
        if confidence:
            props[f"{n}_confidence"] = {"type": "number"}
            req.append(f"{n}_confidence")
    return {"type": "object", "additionalProperties": False, "properties": props, "required": req}
