"""Small, dependency-light metric helpers, used by analyze.py to score one system's results.

Every function here takes plain lists/arrays (confidence values, booleans) and returns a
plain float or list - no knowledge of cases, arms or the rest of the benchmark's data model,
so these are easy to unit-test in isolation (see tests/test_metrics.py) and easy to reuse
outside this repo if you just want the calibration math.
"""
import math

import numpy as np


def wilson(k, n, z=1.96):
    """95% Wilson interval for a proportion. Returns (lo, hi); (nan, nan) if n == 0."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def reliability_bins(conf, correct, n_bins=10):
    """Equal-width bins on [0.5, 1] would hide low-confidence mass, so use [0, 1]."""
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    idx = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    bins = []
    for b in range(n_bins):
        m = idx == b
        if m.any():
            bins.append({"lo": b / n_bins, "hi": (b + 1) / n_bins, "n": int(m.sum()),
                         "conf": float(conf[m].mean()), "acc": float(correct[m].mean())})
    return bins


def ece(conf, correct, n_bins=10):
    """Expected calibration error: weighted mean |accuracy - confidence| over bins."""
    conf = np.asarray(conf, float)
    if len(conf) == 0:
        return float("nan")
    return float(sum(b["n"] * abs(b["acc"] - b["conf"]) for b in reliability_bins(conf, correct, n_bins)) / len(conf))


def brier(conf, correct):
    """Brier score: mean squared error between confidence and correctness (0 or 1). Like ECE,
    lower is better and 0 is perfect, but it penalizes big misses more (squared, not absolute)
    and doesn't depend on how the values are binned."""
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    return float(np.mean((conf - correct) ** 2)) if len(conf) else float("nan")


def auroc_error_detection(conf, correct):
    """P(random correct decision has higher confidence than random wrong one).
    0.5 = confidence carries no information about errors. nan if only one class present."""
    conf, correct = np.asarray(conf, float), np.asarray(correct, bool)
    pos, neg = conf[correct], conf[~correct]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    gt = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(gt / (len(pos) * len(neg)))


def risk_coverage(conf, correct):
    """For every distinct confidence value seen, as if it were used as an auto-approve
    threshold: what share of cases would clear it (coverage), and what's the accuracy among
    the ones that do (selective accuracy)? This is the data behind the confidence-routing
    chart (visuals.py's routing()) and the auto-approve policy simulation in analyze.py.
    -> list of (coverage, accuracy, threshold), one entry per distinct confidence value, in
    ascending threshold order (so coverage is non-increasing along the list)."""
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    out = []
    for t in sorted(set(conf.tolist())):
        m = conf >= t
        out.append((float(m.mean()), float(correct[m].mean()), float(t)))
    return out


def percentile(xs, q):
    """np.percentile, but tolerant of None/nan entries (a failed call has no latency) and of
    an empty list (returns nan rather than raising)."""
    xs = [x for x in xs if x is not None and not math.isnan(x)]
    return float(np.percentile(xs, q)) if xs else float("nan")


def cluster_bootstrap_ci(case_values, n_boot=10000, seed=0, alpha=0.05):
    """A 95% CI for a mean, via a nonparametric bootstrap that resamples at the CLUSTER level.

    Use this instead of wilson() whenever the underlying observations are not independent -
    e.g. this benchmark calls each case 3 times (repeats), and those 3 calls are correlated
    (a case that's hard for a system tends to be hard on every repeat, not independently hard
    each time). Pooling all repeats into one binomial-style interval (wilson) treats 300
    correlated rows as 300 independent ones, which understates the true uncertainty - the
    interval looks narrower than it should.

    `case_values` is one summary value PER CLUSTER (per case), already collapsed across that
    case's own repeats (e.g. the fraction of its repeats that were correct: 0, 1/3, 2/3 or 1
    for a 3-repeat case). The bootstrap resamples cases with replacement, so it reflects
    between-case variability, which is the variability actually of interest ("would a
    different set of 100 cases have given a similar answer?"), not within-case repeat noise.
    """
    vals = np.asarray(case_values, float)
    n = len(vals)
    if n == 0:
        return float("nan"), float("nan")
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, n, size=(n_boot, n))
    boot_means = vals[idx].mean(axis=1)
    lo, hi = np.percentile(boot_means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def paired_bootstrap_diff(values_a, values_b, n_boot=10000, seed=0, alpha=0.05):
    """Compares two systems evaluated on the SAME matched cases (a paired design), via a
    paired bootstrap: every resample draws the same case indices for both systems, so
    whatever correlation exists between them (e.g. both systems tend to fail on the same hard
    cases) is preserved rather than washed out - the correct way to ask "is system A actually
    better than system B," as opposed to eyeballing whether their two separate, unpaired
    confidence intervals happen to overlap (which is neither necessary nor sufficient for a
    real difference).

    `values_a`/`values_b` are the same per-case summary values cluster_bootstrap_ci takes, for
    the same cases in the same order. Returns (point_estimate, lo, hi, two_sided_p) where
    point_estimate = mean(a) - mean(b), (lo, hi) is its 95% CI, and two_sided_p is a bootstrap
    p-value for "the true difference is zero" (twice the smaller tail's share of resamples
    that crossed zero, capped at 1)."""
    a, b = np.asarray(values_a, float), np.asarray(values_b, float)
    n = len(a)
    assert len(b) == n, "paired comparison needs the same number of cases for both systems"
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, n, size=(n_boot, n))
    diffs = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    point = float(a.mean() - b.mean())
    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    # Both tails computed directly (not as 1 - the other) because they can share probability
    # mass exactly at zero (e.g. two identical systems resample to diff=0 every time) - using
    # 1 - p_le as "the other tail" would then be wrong by double-counting that mass.
    p_le = float((diffs <= 0).mean())
    p_ge = float((diffs >= 0).mean())
    p_two_sided = min(2 * min(p_le, p_ge), 1.0)
    return point, float(lo), float(hi), p_two_sided
