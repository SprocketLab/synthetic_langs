#!/usr/bin/env python3
"""Slope chart (Sec. 6): perplexity gap to the best architecture in each of two languages, window 20,
on a shared axis with 0% at the top. Crossing lines are rank swaps.

Values are medians over seeds of the best validation PPL (5 seeds on SL2 / sl-eng, 3 on BabyLM).

    python plot_rank_slope.py                              # sl-eng vs BabyLM -> sleng_babylm.png
    python plot_rank_slope.py --pair sl2 sleng             # -> sl2_sleng.png
    python plot_rank_slope.py --pair sl2 babylm --babylm prepretrain/none_lrfix --name sl2_babylm_lrfix
"""
import os, json, glob, argparse, itertools
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import paper_style as P

_HERE = os.path.dirname(os.path.abspath(__file__))
SYN = os.path.join(_HERE, "..", "..", "results", "synthetic")
OUT = os.path.join(_HERE, "..", "..", "figures")
os.makedirs(OUT, exist_ok=True)
CONFIGS = ["pure_tf", "4s4t_alternate", "6s2t_start", "6s2t_even", "7s1t_mid", "pure_ssm",
           "pure_dp", "4d4t_alternate", "6d2t_start"]


def best_ppls(d, cfg, key):
    vals = []
    for p in glob.glob(os.path.join(d, cfg, "*seed*.json")):
        r = json.load(open(p))
        vals.append(min(r["val_ppls"]) if key == "val_ppls" else r[key])
    return np.array(vals)


def spread(y, gap):
    """Spread label positions so neighbours are at least `gap` apart, keeping their order."""
    order, pos = np.argsort(y), np.array(y, dtype=float)
    for _ in range(300):
        moved = False
        for a, b in zip(order[:-1], order[1:]):
            if pos[b] - pos[a] < gap:
                d = (gap - (pos[b] - pos[a])) / 2
                pos[a] -= d; pos[b] += d; moved = True
        if not moved:
            break
    return pos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", nargs=2, default=["sleng", "babylm"], choices=["sl2", "sleng", "babylm"])
    ap.add_argument("--sleng", default="lang_sl2_sweep/sl-eng-d04_w20_n100k")
    ap.add_argument("--babylm", default="prepretrain/none", help="BabyLM results directory under results/synthetic")
    ap.add_argument("--name", default=None, help="output name (default <left>_<right>)")
    a = ap.parse_args()
    a.name = a.name or "_".join(a.pair)
    P.apply()
    # language -> (axis label, results dir, field)
    lang = {"sl2": ("SL2", "lang_sl2_w20", "val_ppls"),
            "sleng": ("sl-eng", a.sleng, "val_ppls"),
            "babylm": ("BabyLM", a.babylm, "best_ppl")}
    (lab_l, dir_l, key_l), (lab_r, dir_r, key_r) = lang[a.pair[0]], lang[a.pair[1]]

    rows = []
    for c in CONFIGS:
        l = best_ppls(os.path.join(SYN, dir_l), c, key_l)
        r = best_ppls(os.path.join(SYN, dir_r), c, key_r)
        if len(l) and len(r):
            rows.append((c, np.median(l), np.median(r)))

    # PPL gap to the best architecture (%), 0 at the top
    best_l, best_r = min(r[1] for r in rows), min(r[2] for r in rows)
    rows = [(c, 100 * (vl / best_l - 1), 100 * (vr / best_r - 1)) for c, vl, vr in rows]
    gmax = max(max(r[1], r[2]) for r in rows)
    top = gmax * 1.04
    f = lambda g: 1 - g / top                       # y in [0, 1], 1 = best

    fig, ax = plt.subplots(figsize=(P.ACL_COL, 2.55))
    fig.subplots_adjust(left=0.0, right=1.0, top=0.97, bottom=0.09)
    xl, xr = 0.0, 1.0
    for x in (xl, xr):
        ax.plot([x, x], [0, 1], color=P.GUIDE, lw=1.2, zorder=1, solid_capstyle="round")
    for c, gl, gr in rows:
        ax.plot([xl, xr], [f(gl), f(gr)], color=P.color(c), lw=2.0, zorder=2, solid_capstyle="round")
        ax.plot([xl, xr], [f(gl), f(gr)], ls="none", color=P.color(c), zorder=3, **P.MARKER)

    # labels right of the second axis
    ys0 = [f(r[2]) for r in rows]
    for (c, *_), y in zip(rows, spread(ys0, 0.083)):
        ax.text(xr + 0.06, y, P.LAYOUT[c], ha="left", va="center", fontsize=6.6,
                family="DejaVu Sans Mono", fontweight="bold", color=P.color(c))

    # gap ticks on both axes, labelled on the left one
    for x in (xl, xr):
        for t in np.arange(0, top + 1e-9, 5):
            ax.plot([x - 0.025, x], [f(t)] * 2, color=P.GUIDE, lw=1.2, solid_capstyle="butt", zorder=1)
            if x == xl:
                ax.text(x - 0.04, f(t), f"{t:g}", ha="right", va="center", fontsize=6.3, color="#555555")

    # Kendall's tau between the two rankings
    tau = np.mean([np.sign(p[1] - q[1]) * np.sign(p[2] - q[2]) for p, q in itertools.combinations(rows, 2)])
    ax.text(0.5, -0.08, f"τ = {tau:.2f}".replace("-", "−"), ha="center", va="center", fontsize=6.6,
            fontstyle="italic", color="#555555")

    ax.set_xlim(-0.86, 1.86)   # symmetric margins
    ax.set_ylim(-0.13, 1.04)
    ax.set_xticks([xl, xr], [lab_l, lab_r])
    ax.set_yticks([])
    for s in ("left", "bottom"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="x", length=0, pad=4)
    ax.text(xl - 0.30, 0.5, "PPL gap to best (%)", rotation=90, ha="center", va="center", fontsize=7.5)

    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, f"{a.name}.png"))   # no tight bbox, keeps the margins
    print("saved", os.path.join(OUT, a.name + ".png"))


if __name__ == "__main__":
    main()
