"""Renders LinkedIn-ready PNGs from metrics.json (no API calls).

    python visuals.py [--dir results]   ->  <dir>/figures/*.png

Design: one fixed color per system across every figure (identity, not rank), direct labels,
recessive grid, ink in text tokens rather than series colors. Every metric name carries an
explicit up/down arrow so "higher/lower is better" never has to be inferred. Dry-run output
is watermarked.
"""
import argparse
import json
import math
import textwrap
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                     "font.family": "DejaVu Sans", "text.color": INK, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID, "axes.grid": False})
CAT_LABEL = {"clear": "Clear", "ambiguous": "Ambiguous", "missing_evidence": "Missing\nevidence",
             "conflicting_evidence": "Conflicting\nevidence", "adversarial": "Adversarial"}
PCT = lambda v, _: f"{v*100:.0f}%"


def ok(x):
    return x is not None and not (isinstance(x, float) and math.isnan(x))


def place_labels(ax, fig, points, fontsize=12, marker_radius_pt=9, avoid_artists=None, pad_pt=5):
    """Places one text label per (x, y, text, color) point with no overlaps, checking real rendered
    text bounding boxes (not just distance between the dots) against every already-placed label,
    every marker, and any other artist passed via avoid_artists (e.g. a fixed annotation elsewhere
    on the axes). A fixed offset direction is not enough: a long label ("gpt-5.6-sol (3 calls)") can
    reach clear past a neighboring dot even when the two dots themselves aren't close together.
    Padding is a fixed number of points, not a percentage - a percentage of a short bbox (a 2-word
    label) rounds to near-zero and lets labels visually touch even when "no overlap" is reported."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    pad_px = pad_pt * fig.dpi / 72
    r_px = marker_radius_pt * fig.dpi / 72

    def padded(bb):
        return plt.matplotlib.transforms.Bbox.from_extents(bb.x0 - pad_px, bb.y0 - pad_px, bb.x1 + pad_px, bb.y1 + pad_px)

    placed = []
    for x, y, *_ in points:
        px, py = ax.transData.transform((x, y))
        placed.append(plt.matplotlib.transforms.Bbox.from_bounds(px - r_px, py - r_px, 2 * r_px, 2 * r_px))
    if avoid_artists:
        placed += [padded(a.get_window_extent(renderer)) for a in avoid_artists]
    candidates = [(10, 8, "left", "bottom"), (10, -8, "left", "top"), (-10, 8, "right", "bottom"),
                  (-10, -8, "right", "top"), (0, 16, "center", "bottom"), (0, -16, "center", "top")]
    for x, y, text, color in sorted(points, key=lambda p: p[0]):
        chosen = None
        for dx, dy, ha, va in candidates:
            t = ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points",
                             ha=ha, va=va, fontsize=fontsize, color=INK, clip_on=False)
            fig.canvas.draw()
            bb = padded(t.get_window_extent(renderer))
            if not any(bb.overlaps(p) for p in placed):
                chosen = (t, bb)
                break
            t.remove()
        if chosen is None:  # every slot collided (crowded plot) - keep the first guess anyway
            dx, dy, ha, va = candidates[0]
            t = ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points",
                             ha=ha, va=va, fontsize=fontsize, color=INK, clip_on=False)
            fig.canvas.draw()
            chosen = (t, padded(t.get_window_extent(renderer)))
        placed.append(chosen[1])


def cal_quality(cal):
    """One-word verdict on a system's confidence, not just the raw ECE number."""
    if not cal:
        return None
    if cal["ece"] < 0.05:
        return "well-calibrated"
    return "overconfident" if cal["mean_confidence"] > cal["accuracy"] else "underconfident"


def finish(fig, res, path, title, subtitle, note=None):
    fig.text(0.05, 0.965, title, fontsize=21, fontweight="bold", va="top", color=INK)
    fig.text(0.05, 0.925, subtitle, fontsize=12.5, va="top", color=INK2)
    meta = res.get("meta", {})
    src = f"{res['n_cases']} synthetic cases · {', '.join(a['label'] for a in res['arms'].values() if a['system'] in ('jev', 'llm'))}"
    if meta.get("date"):
        src += f" · run {meta['date']}"
    foot = f"{note}\n{src}" if note else src
    fig.text(0.05, 0.04, foot, fontsize=9.5, color=MUTED, va="bottom", linespacing=1.5)
    if res.get("dry_run"):
        fig.text(0.5, 0.5, "DRY RUN\nSIMULATED DATA", fontsize=64, color="#d03b3b", alpha=0.16,
                 rotation=28, ha="center", va="center", fontweight="bold")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print("wrote", path)


def hero_scorecard(res, out):
    """The lead image: one row per metric, one column per system. Each row tuple is:
      (name, subtitle, get(m) -> value, format(value) -> str, higher_is_better,
       ci(m) -> (lo, hi) | None, [subtext(m) -> str | None], [api_only])
    `ci` prints a confidence interval under the value (used only for the Accuracy row).
    The optional 7th element is a per-cell caption under the value (e.g. the calibration
    verdict, or the auto-approve sample size); the optional 8th, if present and truthy,
    restricts "best in row" shading to the API-based systems (jev/llm) - used for latency and
    cost, where the offline baselines are trivially faster/cheaper and not a fair comparison."""
    arms = [(k, m) for k, m in res["arms"].items() if m["system"] != "llm_split"]
    rows = [
        ("Accuracy ↑", "all 3 decisions correct", lambda m: m["accuracy"]["all_three"]["rate"], lambda v: f"{v*100:.0f}%", True,
         lambda m: (m["accuracy"]["all_three"]["case_bootstrap"]["lo"], m["accuracy"]["all_three"]["case_bootstrap"]["hi"])),
        ("Calibration ↓", "ECE: lower = more trustworthy*", lambda m: (m["calibration"] or {}).get("ece"), lambda v: f"{v:.2f}", False,
         None, lambda m: cal_quality(m["calibration"])),
        ("Missed escalations ↓", "needed a human, wasn't flagged", lambda m: m["manual_routing_confusion"]["missed_escalation_rate"], lambda v: f"{v*100:.0f}%", False, None),
        ("Unneeded escalations ↓", "flagged a human, didn't need one", lambda m: m["manual_routing_confusion"]["unneeded_escalation_rate"], lambda v: f"{v*100:.0f}%", False, None),
        ("p95 latency ↓", "seconds per case", lambda m: m["latency_s"]["p95"], lambda v: f"{v:.2f} s" if v >= 0.01 else "<0.01 s", False, None, None, True),
        ("Cost ↓", "per 1,000 decisions", lambda m: m["cost_per_1k_decisions"], lambda v: f"${v:.2f}" if v >= 0.005 else "~$0", False, None, None, True),
    ]
    fig = plt.figure(figsize=(10.8, 10.8))
    ax = fig.add_axes([0.05, 0.16, 0.9, 0.64]); ax.set_axis_off(); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    lab_w = 0.30
    cw = (1 - lab_w) / len(arms)
    for j, (_, m) in enumerate(arms):
        x = lab_w + cw * (j + .5)
        ax.add_patch(plt.Circle((x, 0.975), 0.012, color=m["color"], transform=ax.transData, clip_on=False))
        ax.text(x, 0.945, textwrap.fill(m["label"], 11), ha="center", va="top", fontsize=12.5, fontweight="bold", linespacing=1.15)
    rh = 0.86 / len(rows)
    for i, (name, sub, get, fmt, hib, ci, *extra) in enumerate(rows):
        subtxt = extra[0] if extra else None
        api_only = len(extra) > 1
        y = 0.86 - rh * (i + 1)
        ax.plot([0, 1], [y + rh, y + rh], color=GRID, lw=1)
        ax.text(0.0, y + rh / 2 + 0.012, name, fontsize=15, fontweight="bold", va="center")
        ax.text(0.0, y + rh / 2 - 0.028, sub, fontsize=10.5, color=MUTED, va="center")
        vals = [get(m) for _, m in arms]
        cand = [v for (k, m), v in zip(arms, vals) if ok(v) and (not api_only or m["system"] in ("jev", "llm"))]
        best = (max if hib else min)(cand) if cand else None
        for j, ((_, m), v) in enumerate(zip(arms, vals)):
            x = lab_w + cw * (j + .5)
            if not ok(v):
                yy = y + rh / 2 + (0.012 if subtxt and subtxt(m) else 0)
                ax.text(x, yy, "n/a", ha="center", va="center", fontsize=13, color=MUTED)
                if subtxt and subtxt(m):
                    ax.text(x, y + rh / 2 - 0.03, subtxt(m), ha="center", va="center", fontsize=9.5, color=MUTED)
                continue
            is_best = v == best and len(cand) > 1 and (not api_only or m['system'] in ('jev', 'llm'))
            if is_best:
                ax.add_patch(FancyBboxPatch((x - cw * .42, y + rh * .16), cw * .84, rh * .68, boxstyle="round,pad=0,rounding_size=0.012",
                                            fc="#eaf1fb", ec="none", zorder=0))
            ax.text(x, y + rh / 2 + (0.012 if (ci or subtxt) else 0), fmt(v), ha="center", va="center", fontsize=19 if is_best else 17,
                    fontweight="bold" if is_best else "normal")
            if subtxt and subtxt(m):
                ax.text(x, y + rh / 2 - 0.03, subtxt(m), ha="center", va="center", fontsize=9.5, color=MUTED)
            if ci:
                lo, hi = ci(m)
                ax.text(x, y + rh / 2 - 0.03, f"{lo*100:.0f}-{hi*100:.0f}%", ha="center", va="center", fontsize=9.5, color=MUTED)
    finish(fig, res, out / "1_hero_scorecard.png", "Jev vs LLMs vs classic baselines",
           "Enterprise agent-routing decisions: which agent, does routing need a person?, high risk?",
           note="↑ higher is better, ↓ lower is better. Shaded = best in row (among comparable systems). Accuracy's 95% CI is a\n"
                "cluster bootstrap by case (wider but more honest than pooling the 3 correlated repeats per case as if\n"
                "independent). *ECE (Calibration) = 0 means stated confidence exactly matches real accuracy; it's a size only,\n"
                "not a direction (a system can over- or under-sell itself at the same ECE) - see the reliability chart.")


def category_heatmap(res, out):
    """Rows = systems, columns = case categories (difficulty buckets), cell = that system's
    all-three-correct accuracy on just that category. Z is that grid as a plain numpy array so
    imshow can render it directly and nanargmax can find each column's best cell in one call."""
    arms = list(res["arms"].values())
    cats = list(res["category_counts"])
    fig = plt.figure(figsize=(10.8, 8.8))
    ax = fig.add_axes([0.2, 0.14, 0.76, 0.64])
    Z = np.array([[m["by_category"].get(c, {}).get("rate", float("nan")) for c in cats] for m in arms])
    ramp = ["#cde2fb", "#9ec5f4", "#5598e7", "#256abf", "#104281"]  # sequential blue, light -> dark
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq", ramp)
    ax.imshow(Z, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            v = Z[i, j]
            ax.text(j, i, f"{v*100:.0f}%", ha="center", va="center", fontsize=15, fontweight="bold",
                    color="#ffffff" if v > .55 else INK)
    # Outline the best-scoring system in each column so "which system wins this workload type"
    # is answerable at a glance, without reading every number in the column.
    best_per_col = np.nanargmax(Z, axis=0)
    for j, i in enumerate(best_per_col):
        ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=INK, linewidth=3, zorder=6))
    ax.set_xticks(range(len(cats)), [f"{CAT_LABEL.get(c, c)}\n(n={res['category_counts'][c]})" for c in cats], fontsize=11)
    ax.set_yticks(range(len(arms)), [m["label"] for m in arms], fontsize=12.5)
    ax.xaxis.tick_top(); ax.tick_params(length=0)
    for s in ax.spines.values(): s.set_visible(False)
    for k in range(1, Z.shape[0]):
        ax.axhline(k - .5, color=SURFACE, lw=3)
    for k in range(1, Z.shape[1]):
        ax.axvline(k - .5, color=SURFACE, lw=3)
    finish(fig, res, out / "2_accuracy_by_category.png", "Where does each approach break?",
           "Share of cases with all three decisions correct, by workload type",
           note="Outlined cell = best system for that workload type.")


def reliability(res, out):
    """One reliability diagram per system with confidence values (skips llm_split, and any
    arm with no confidence at all, like keyword rules). Each point is one of metrics.py's
    reliability_bins: x = that bin's mean stated confidence, y = that bin's actual accuracy."""
    arms = [m for m in res["arms"].values() if m["calibration"] and m["system"] != "llm_split"]
    if not arms:
        return
    n = len(arms)
    fig, axes = plt.subplots(1, n, figsize=(max(10.8, 3.6 * n), 6.9), squeeze=False)
    fig.subplots_adjust(left=.07, right=.97, top=.77, bottom=.21, wspace=.28)
    xx = np.linspace(0, 1, 2)  # just the two endpoints - fill_between only needs a straight line here
    for ax, m in zip(axes[0], arms):
        cal = m["calibration"]
        # Below the diagonal the system claims more confidence than it earns (overconfident);
        # above it, the opposite (underconfident). Very light fills so the data stays legible.
        ax.fill_between(xx, xx, 0, color="#d03b3b", alpha=0.045, zorder=0)
        ax.fill_between(xx, xx, 1, color="#2a78d6", alpha=0.045, zorder=0)
        ax.plot([0, 1], [0, 1], color=GRID, lw=1.5, ls=(0, (4, 3)), zorder=1)
        b = cal["bins"]
        ax.plot([x["conf"] for x in b], [x["acc"] for x in b], color=m["color"], lw=2, zorder=2)
        ax.scatter([x["conf"] for x in b], [x["acc"] for x in b], s=[20 + 700 * x["n"] / cal["n"] for x in b],
                   color=m["color"], edgecolor=SURFACE, linewidth=1.5, zorder=3)
        ax.set_xlim(0.0, 1.02); ax.set_ylim(-0.02, 1.02)
        ax.set_title(m["label"], fontsize=13, fontweight="bold", loc="left")
        ax.text(0.02, 0.03, f"ECE {cal['ece']:.2f} · {cal_quality(cal)}", fontsize=11, color=INK)
        ax.set_xlabel("stated confidence"); ax.set_ylabel("actual accuracy" if ax is axes[0][0] else "")
        ax.grid(color=GRID, lw=.8); ax.set_axisbelow(True)
        for s in ("top", "right"): ax.spines[s].set_visible(False)
    axes[0][0].text(0.02, 0.97, "underconfident zone", fontsize=8.5, color="#1c5cab", va="top")
    axes[0][0].text(0.98, 0.05, "overconfident zone", fontsize=8.5, color="#a83232", ha="right", va="bottom")
    finish(fig, res, out / "3_reliability.png", "Does confidence mean anything?",
           "ECE = the average gap between stated confidence and real accuracy (0 = perfectly trustworthy, higher = less so).\n"
           "A point on the dashed diagonal means confidence matches accuracy exactly.",
           note="Below the line (red) = overconfident: claims more certainty than it earns. Above (blue) = underconfident:\n"
                "more accurate than it lets on. Dot size = how many decisions fall in that confidence bucket. ECE is a size\n"
                "only, not a direction, so two systems can share a number while erring in opposite directions (labels above).")


def routing(res, out):
    """Two panels sharing an x-axis (the confidence bar for auto-approving a case), ascending
    from 0 (approve everything) to 1 (approve nothing): top = accuracy among the cases that
    clear the bar, bottom = what share of all cases that is. Colored bands mark the same
    3-tier routing policy (human / deeper reasoning / auto-approve) analyze.py simulates."""
    arms = [m for m in res["arms"].values() if m["routing"] and m["system"] != "llm_split"]
    if not arms:
        return
    pol0 = arms[0]["routing"]["policy"]
    LO, HI = pol0["lo"], pol0["hi"]
    MIN_COV = 0.03  # stop each line once fewer than ~3% of cases still qualify (too little support to trust)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10.8, 9.4), sharex=True,
                                    gridspec_kw={"height_ratios": [2, 1], "hspace": 0.10})
    fig.subplots_adjust(left=.10, right=.80, top=.76, bottom=.19)
    zones = [(0, LO, "#f6c6c2", "human review"), (LO, HI, "#fbdfa3", "deeper reasoning"), (HI, 1.0, "#c3e6c3", "auto-approve")]
    for ax in (ax1, ax2):
        for x0, x1, color, _ in zones:
            ax.axvspan(x0, x1, color=color, alpha=0.4, zorder=0)
    zone_labels = [ax1.text((x0 + x1) / 2, 1.09, label if x1 - x0 >= 0.2 else "auto", ha="center", va="bottom",
                            fontsize=9.5, color=INK2) for x0, x1, _, label in zones]

    end_points_top = []
    for m in arms:
        pts = [p for p in m["routing"]["curve"] if p[0] >= MIN_COV]  # (coverage, accuracy, threshold), threshold ascending
        if not pts:
            continue
        thr, acc, cov = [p[2] for p in pts], [p[1] for p in pts], [p[0] for p in pts]
        ax1.plot(thr, acc, color=m["color"], lw=2.4, solid_joinstyle="round")
        ax2.plot(thr, cov, color=m["color"], lw=2.4, solid_joinstyle="round")
        end_points_top.append((thr[-1], acc[-1], m["label"], m["color"]))

    ax1.set_xlim(0, 1); ax1.set_ylim(0, 1.02); ax2.set_ylim(0, 1.02)
    ax1.yaxis.set_major_formatter(PCT); ax2.yaxis.set_major_formatter(PCT); ax2.xaxis.set_major_formatter(PCT)
    ax1.set_ylabel("accuracy\namong cases ≥ bar", fontsize=10.5)
    ax2.set_ylabel("share of cases\nthat clear the bar", fontsize=10.5)
    ax2.set_xlabel("confidence bar for auto-approving a case", fontsize=10.5)
    for ax in (ax1, ax2):
        ax.grid(color=GRID, lw=.8); ax.set_axisbelow(True)
        for s in ("top", "right"): ax.spines[s].set_visible(False)
    # Direct end-labels work on the top panel, but the bottom panel's lines all converge toward
    # zero at the right edge - end-labeling it collides lines/labels/tick-labels in that corner.
    # Both panels share one color per system, so a single legend (not per-panel labels) covers both.
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=m["color"], lw=3, label=m["label"]) for m in arms]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.45, 0.855), ncol=len(handles),
               frameon=False, fontsize=10.5, handlelength=1.4, columnspacing=1.3)
    place_labels(ax1, fig, end_points_top, fontsize=11, marker_radius_pt=0, avoid_artists=zone_labels)
    finish(fig, res, out / "4_confidence_routing.png", "Where should the confidence bar sit?",
           "Top: accuracy on the cases each system is confident enough to auto-approve. Bottom: what share of all\n"
           "cases that is: same colors as above; both curves fall as the bar rises.",
           note="Bands = the 3-tier policy from the write-up: human <60%, deeper reasoning 60-90%, auto-approve ≥90%.")


def latency_cost(res, out):
    """Log-log scatter of median latency vs. cost per 1,000 decisions, API-based systems only
    (the offline baselines round to ~$0 and ~0s, which would break the log scale and isn't a
    fair comparison anyway - see the note in the chart's own footer)."""
    arms = [m for m in res["arms"].values() if m["system"] in ("jev", "llm", "llm_split") and ok(m["cost_per_1k_decisions"])]
    if not arms:
        return
    fig = plt.figure(figsize=(10.8, 7.4))
    ax = fig.add_axes([0.12, 0.22, 0.82, 0.54])
    points = []
    for m in arms:
        x, y = m["latency_s"]["p50"], max(m["cost_per_1k_decisions"], 1e-3)
        ax.scatter([x], [y], s=260, color=m["color"], edgecolor=SURFACE, linewidth=2, zorder=3)
        points.append((x, y, m["label"], m["color"]))
    ax.set_xscale("log"); ax.set_yscale("log")
    lo, hi = min(m["latency_s"]["p50"] for m in arms), max(m["latency_s"]["p50"] for m in arms)
    ax.set_xticks([t for t in (.1, .2, .3, .5, 1, 2, 3, 5, 10, 20, 30) if lo / 2.5 <= t <= hi * 2.5])
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:g} s"); ax.xaxis.set_minor_formatter(lambda v, _: "")
    ax.yaxis.set_major_formatter(lambda v, _: f"${v:g}"); ax.yaxis.set_minor_formatter(lambda v, _: "")
    ax.set_xlabel("median latency per case → slower", fontsize=11)
    ax.set_ylabel("cost per 1,000 decisions → pricier", fontsize=11)
    ax.grid(color=GRID, lw=.8, which="major"); ax.set_axisbelow(True)
    ax.margins(x=.3, y=.3)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    place_labels(ax, fig, points)

    jevm = next((m for m in arms if m["system"] == "jev"), None)
    llms = [m for m in arms if m["system"] in ("llm", "llm_split")]
    if jevm and llms:
        cheapest = min(llms, key=lambda m: m["cost_per_1k_decisions"])
        fastest = min(llms, key=lambda m: m["latency_s"]["p50"])
        cost_x = cheapest["cost_per_1k_decisions"] / jevm["cost_per_1k_decisions"]
        lat_x = fastest["latency_s"]["p50"] / jevm["latency_s"]["p50"]
        ax.text(0.02, 0.97, f"Jev is ~{lat_x:.0f}x faster and ~{cost_x:.0f}x cheaper\nthan the fastest/cheapest LLM tested here",
                transform=ax.transAxes, fontsize=10.5, va="top", ha="left", color=INK,
                bbox=dict(boxstyle="round,pad=0.5", fc="#eaf1fb", ec="none"))
    finish(fig, res, out / "5_latency_vs_cost.png", "What does a decision cost?",
           "Lower-left is better. Offline baselines (~$0, ~0 s) are omitted.",
           note="Both axes are logarithmic: each gridline is 10x the previous, not an equal step.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=Path(__file__).parent / "results")
    a = ap.parse_args()
    res = json.loads((a.dir / "metrics.json").read_text())
    out = a.dir / "figures"; out.mkdir(exist_ok=True)
    for f in (hero_scorecard, category_heatmap, reliability, routing, latency_cost):
        f(res, out)


if __name__ == "__main__":
    main()
