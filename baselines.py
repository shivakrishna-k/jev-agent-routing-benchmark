"""Non-LLM baselines, evaluated offline on the same cases.

rules    : hand-written keyword rules (no confidence)
tfidf_lr : TF-IDF + logistic regression, 5-fold stratified cross-validation so every case is
           predicted by a model that never saw it. predict_proba supplies a confidence.
"""
import re
import time

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

import spec

# Substring keyword lists for the "keyword rules" baseline. Deliberately naive: this baseline
# exists to show what a system with no understanding of context does with this dataset,
# including on the adversarial cases designed to exploit exactly this kind of matching
# (e.g. "please summarize this trade SURVEILLANCE policy" reads as a surveillance case to it).
KW = {
    "human": ["human", "investigator", "someone to investigate", "whistleblower", "grievance", "senior person", "lawyer", "a person"],
    "trade_surveillance": ["surveillance", "alert", "spoof", "layering", "wash", "order", "cancellation", "front-run", "suspicious", "unusual trading", "trading activity", "manipulat", "market abuse"],
    "market_risk": ["exposure", "limit", "var ", "stress", "sensitiv", "position", "shock", "delta", "risk appetite", "curve"],
    "compliance": ["policy", "regulat", "compliance", "retention", "governance", "control", "standard", "attest"],
}
HR_KW = ["human", "investigator", "final decision", "sign off", "sign-off", "person", "contradict", "conflict", "incomplete", "missing", "not available", "who decides", "sign"]
RISK_KW = ["serious", "breach", "suspicious", "unusual", "spoof", "conduct", "significant", "regulator", "limit", "stress", "exceed", "manipulat", "whistle", "abuse"]
# Tie-break order when two agents score the same number of keyword hits (e.g. 0-0): "human"
# wins ties first since a missed human-needed case is the worse failure mode of the four.
PRIORITY = ["human", "trade_surveillance", "market_risk", "compliance"]


def _has(t, words):
    """Count how many of `words` appear as substrings in `t` (already lowercased)."""
    return sum(w in t for w in words)


def rules_predict(request):
    """Agent = whichever category's keyword list matches most words in the request, ties
    broken by PRIORITY. manual_routing_required/high_risk are independent yes/no keyword
    checks - unlike the other systems, this baseline has no shared reasoning between the
    three questions."""
    t = request.lower()
    scores = {a: _has(t, KW[a]) for a in PRIORITY}
    best = max(PRIORITY, key=lambda a: (scores[a], -PRIORITY.index(a)))
    return {"agent": (best, None),
            "manual_routing_required": (_has(t, HR_KW) > 0, None),
            "high_risk": (_has(t, RISK_KW) > 0, None)}


def run_rules(cases):
    """Runs the keyword-rules baseline over every case. No confidence value (None): plain
    keyword matching has no notion of "how sure" it is, unlike the other systems compared."""
    rows = []
    for c in cases:
        t0 = time.perf_counter()
        d = rules_predict(c["request"])
        rows.append({"case_id": c["id"], "arm": "rules", "system": "rules", "model": "keywords",
                     "repeat": 0, "latency_s": time.perf_counter() - t0, "decisions": d, "usage": {"cost": 0.0}})
    return rows


def run_tfidf(cases, folds=5, seed=0):
    """TF-IDF + logistic regression baseline, cross-validated so every case is scored by a
    model that never trained on it (otherwise the classifier would just be memorizing the 100
    cases, not generalizing - meaningless as an accuracy number).

    One vectorizer + one classifier per fold, per question (agent / manual_routing_required /
    high_risk): fit on the fold's training split only, predict on its held-out test split, so
    across all 5 folds every case gets exactly one prediction per question, always from a
    model that hadn't seen it. predict_proba's max-class probability is used as this
    baseline's confidence.
    """
    X = [c["request"] for c in cases]
    ys = {"agent": [c["expected_agent"] for c in cases],
          "manual_routing_required": [c["manual_routing_required"] for c in cases],
          "high_risk": [c["high_risk"] for c in cases]}
    preds = [dict() for _ in cases]
    skf = StratifiedKFold(folds, shuffle=True, random_state=seed)
    # Stratify by (agent, category) so each fold's test split has a representative mix of
    # difficulty levels, not just agent labels; categories too rare for 5-fold stratification
    # (fewer than `folds` cases) fall back to stratifying by agent alone for that case.
    strat = [f"{c['expected_agent']}|{c['category']}" for c in cases]
    strat = [s if strat.count(s) >= folds else s.split("|")[0] for s in strat]
    t_pred = 0.0
    for tr, te in skf.split(X, strat):
        # Fit the vectorizer on the TRAINING fold only - fitting on all 100 cases first would
        # leak the test fold's vocabulary into training, inflating accuracy.
        vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)
        Xtr = vec.fit_transform([X[i] for i in tr])
        t0 = time.perf_counter()
        Xte = vec.transform([X[i] for i in te])
        t_pred += time.perf_counter() - t0  # only inference time counts as "latency" for this baseline
        for q, y in ys.items():
            clf = LogisticRegression(max_iter=1000, C=10).fit(Xtr, [y[i] for i in tr])
            p = clf.predict_proba(Xte)
            for row, i in enumerate(te):
                k = int(np.argmax(p[row]))
                # .item() unwraps numpy bool/str scalars to plain Python types so json.dumps
                # (analyze.py, export_excel.py) doesn't choke on them later.
                preds[i][q] = (clf.classes_[k].item() if hasattr(clf.classes_[k], "item") else clf.classes_[k], float(p[row][k]))
    return [{"case_id": c["id"], "arm": "tfidf_lr", "system": "tfidf_lr", "model": "cv5",
             "repeat": 0, "latency_s": t_pred / len(cases), "decisions": preds[i], "usage": {"cost": 0.0}}
            for i, c in enumerate(cases)]
