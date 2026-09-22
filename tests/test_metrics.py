import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import metrics as M
import backends
import spec


def test_wilson_bounds():
    lo, hi = M.wilson(50, 100)
    assert 0.40 < lo < 0.41 and 0.59 < hi < 0.60
    assert M.wilson(0, 0)[0] != M.wilson(0, 0)[0]  # nan
    assert M.wilson(10, 10)[1] == 1.0


def test_ece_perfect_and_overconfident():
    assert M.ece([1.0] * 10, [1] * 10) == 0
    assert abs(M.ece([0.95] * 10, [1] * 5 + [0] * 5) - 0.45) < 1e-9


def test_brier():
    assert M.brier([1, 0], [1, 0]) == 0
    assert M.brier([1, 1], [0, 0]) == 1


def test_auroc():
    assert M.auroc_error_detection([.9, .9, .2], [1, 1, 0]) == 1.0
    assert M.auroc_error_detection([.5, .5], [1, 0]) == 0.5
    assert math.isnan(M.auroc_error_detection([.9], [1]))


def test_risk_coverage_monotone_coverage():
    cur = M.risk_coverage([.5, .7, .9, .95], [0, 1, 1, 1])
    assert cur[0][0] == 1.0 and cur[-1][0] == 0.25
    assert [c[1] for c in cur][-1] == 1.0


def test_cluster_bootstrap_ci():
    # Every "case" identical -> zero between-case variance -> the CI collapses to a point.
    lo, hi = M.cluster_bootstrap_ci([1.0] * 50)
    assert lo == hi == 1.0
    # A 50/50 split over 100 cases: CI should be a real band centered near 0.5, not a point,
    # and (unlike wilson, which only sees the pooled rate) it must actually use case count -
    # fewer cases makes the interval wider.
    vals = [0.0] * 50 + [1.0] * 50
    lo100, hi100 = M.cluster_bootstrap_ci(vals, seed=1)
    assert lo100 < 0.5 < hi100
    lo10, hi10 = M.cluster_bootstrap_ci(vals[:5] + vals[-5:], seed=1)  # same 50/50 mix, n=10
    assert (hi10 - lo10) > (hi100 - lo100)
    # Same seed -> reproducible; different seed -> not identical (it's actually resampling).
    assert M.cluster_bootstrap_ci(vals, seed=7) == M.cluster_bootstrap_ci(vals, seed=7)
    assert M.cluster_bootstrap_ci(vals, seed=7) != M.cluster_bootstrap_ci(vals, seed=8)


def test_paired_bootstrap_diff():
    # Perfectly separated groups (every "case" identical within each system) -> the paired
    # difference is exact on every single resample, so the CI collapses to a point at the
    # true difference and the two-sided p-value is 0 (no resample ever crossed zero).
    point, lo, hi, p = M.paired_bootstrap_diff([1.0] * 20, [0.0] * 20)
    assert point == 1.0 and lo == hi == 1.0 and p == 0.0
    # Identical systems -> zero difference, and the CI must straddle (or sit at) zero.
    same = [0.4, 0.6, 0.5, 1.0, 0.0] * 4
    point0, lo0, hi0, p0 = M.paired_bootstrap_diff(same, same)
    assert point0 == 0.0 and lo0 <= 0.0 <= hi0 and p0 == 1.0


def test_parse_jev_and_llm():
    # Wire key is still "human_review" here on purpose: this is what every already-collected
    # raw.jsonl response looks like, from before manual_routing_required was renamed. The
    # parser must keep accepting it (see backends.LEGACY_KEYS) and normalize the OUTPUT to
    # today's name, so old data keeps parsing correctly without a re-run.
    body = {"answers": {"agent": {"choice": "compliance", "probabilities": {"compliance": .8, "human": .2}},
                        "human_review": {"noul": .2}, "high_risk": {"noul": .9}}, "usage": {"cost": 1}}
    d, u = backends.parse_jev(body)
    assert d["agent"] == ("compliance", .8) and d["manual_routing_required"] == (False, .8) and d["high_risk"][0] is True
    assert backends.parse_jev({"answers": {}}) is None
    import json
    c = {"agent": "human", "human_review": True, "high_risk": True,
         "agent_confidence": .9, "human_review_confidence": .8, "high_risk_confidence": 1.4}
    d, _ = backends.parse_llm({"choices": [{"message": {"content": json.dumps(c)}}], "usage": {}})
    assert d["manual_routing_required"] == (True, .8) and d["high_risk"][1] == 1.0  # confidence clipped
    assert backends.parse_llm({"choices": [{"message": {"content": "{}"}}]}) is None

    # A response using TODAY's field name (a future run) must also parse correctly.
    c2 = {"agent": "human", "manual_routing_required": False, "high_risk": False,
          "agent_confidence": .7, "manual_routing_required_confidence": .6, "high_risk_confidence": .3}
    d2, _ = backends.parse_llm({"choices": [{"message": {"content": json.dumps(c2)}}], "usage": {}})
    assert d2["manual_routing_required"] == (False, .6)


def test_dataset_invariants():
    cases = spec.load_cases()
    assert len({c["id"] for c in cases}) == len(cases)
    for c in cases:
        assert c["expected_agent"] in c["acceptable_agents"]
        assert set(c["acceptable_agents"]) <= set(spec.AGENTS)
    # labels must not be collinear (the flaw in the first dataset)
    assert any(c["manual_routing_required"] and c["expected_agent"] != "human" for c in cases)
    assert any(c["expected_agent"] == "compliance" and c["high_risk"] for c in cases)
