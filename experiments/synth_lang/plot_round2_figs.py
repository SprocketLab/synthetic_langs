#!/usr/bin/env python3
"""Further paper and appendix figures. Rows are architectures; squares are medians over seeds,
faint lines interquartile ranges.

  orderings       SL2, window 20: best validation PPL and per-sentence recall of all 13 orderings
  nearfar         recall accuracy within and beyond the attention window, from recall_by_distance.json
  prepretrain     BabyLM validation PPL per epoch, from scratch and after SL2 pre-pretraining
  epoch20         BabyLM PPL at epoch 20 with and without SL2 pre-pretraining (slope chart)
  sensitivity_kl  within-grammar and within-non-grammar KL across the SL2 parameter sweep
  sl2_babylm_scatter  best PPL on SL2 against best PPL on BabyLM
  long_training   sl-eng 100k x 15: tau with the SL2 ranking per epoch, and recall per model

    python plot_round2_figs.py [orderings nearfar ... long_training]   # -> figures/<name>.png
"""
import os, sys, json, glob, itertools
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import paper_style as P

_HERE = os.path.dirname(os.path.abspath(__file__))
SYN = os.path.join(_HERE, "..", "..", "results", "synthetic")
OUT = os.path.join(_HERE, "..", "..", "figures")
os.makedirs(OUT, exist_ok=True)
OPEN = {**P.MARKER, "markerfacecolor": "white"}


def style():
    P.apply()
    plt.rcParams["axes.titlesize"] = 6.5


def runs(d, cfg):
    return [json.load(open(p)) for p in sorted(glob.glob(os.path.join(SYN, d, cfg, "*seed*.json")))]


def at_best(r, key, field):
    b = int(np.argmin(r["val_ppls"]))
    return min(r[key], key=lambda e: (abs(e["epoch"] - b), e["epoch"]))[field]


def label_rows(ax, order):
    ax.set_yticks(np.arange(len(order)), [P.LAYOUT[c] for c in order], family="DejaVu Sans Mono", fontsize=6.3)
    for t, c in zip(ax.get_yticklabels(), order):
        t.set_color(P.color(c))
    ax.set_ylim(-0.6, len(order) - 0.4)


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    out = os.path.join(OUT, f"{name}.png")
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
    print("saved", os.path.normpath(out))


# all orderings on SL2

def orderings():
    d = "lang_sl2_w20"
    cfgs = [c for c in P.LAYOUT if runs(d, c)]
    vals = {c: (np.array([min(r["val_ppls"]) for r in runs(d, c)]),
                100 * np.array([at_best(r, "recall_accs", "accuracy_sent") for r in runs(d, c)])) for c in cfgs}
    order = sorted(cfgs, key=lambda c: -np.median(vals[c][0]))   # lowest PPL at the top
    near = json.load(open(os.path.join(SYN, d, "recall_by_distance.json")))["frac_near"]

    fig, axes = plt.subplots(1, 2, figsize=(P.ACL_COL, 2.6), sharey=True)
    fig.subplots_adjust(wspace=0.1)
    for k, (ax, title) in enumerate(zip(axes, ["Best PPL", "Recall (%)"])):
        for y, c in enumerate(order):
            v = vals[c][k]
            ax.plot(np.percentile(v, [25, 75]), [y, y], color=P.color(c), lw=1.6, alpha=0.35, zorder=2,
                    solid_capstyle="butt")   # IQR over seeds
            ax.plot(np.median(v), y, ls="none", color=P.color(c), zorder=3, **{**P.MARKER, "markersize": 4.5})
        ax.set_title(title)
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
    axes[1].axvline(100 * near, color=P.GUIDE, ls="--", lw=1, zorder=1)
    axes[1].text(100 * near + 6, 0.45, "answers only\nnear recalls", ha="left", va="center", fontsize=5.6,
                 color="#555555")
    axes[1].set_xlim(-3, 103)
    axes[0].set_xticks([13.5, 14, 14.5, 15])
    label_rows(axes[0], order)
    save(fig, "sl2_orderings")


# near and far recall

def nearfar():
    settings = [("SL2\n20k sent., 15 ep.", "lang_sl2_w20"),
                ("sl-eng\n20k sent., 15 ep.", "lang_sl2_sweep/sl-eng-d04_w20"),
                ("sl-eng\n100k sent., 15 ep.", "lang_sl2_sweep/sl-eng-d04_w20_n100k"),
                ("sl-eng-g100\n100k sent., 3 ep.", "lang_sl2_sweep/sl-eng-g100_w20_n100k_ep3")]
    order = ["pure_ssm", "7s1t_mid", "6s2t_even", "6s2t_end", "6s2t_start", "4s4t_tf_first", "4s4t_ssm_first",
             "4s4t_alternate", "pure_tf"]   # bottom to top
    fig, axes = plt.subplots(1, len(settings), figsize=(2 * P.ACL_COL, 2.4), sharey=True)
    fig.subplots_adjust(wspace=0.12)
    for ax, (title, d) in zip(axes, settings):
        rbd = json.load(open(os.path.join(SYN, d, "recall_by_distance.json")))
        for y, c in enumerate(order):
            rs = rbd["runs"].get(c)
            if not rs:
                continue
            nr = np.array([r["acc_near"] for r in rs]); fr = np.array([r["acc_far"] for r in rs])
            col = P.color(c)
            # near recalls on the upper sub-row, far on the lower; lines are IQRs over seeds
            for v, dy, mk in ((nr, 0.17, OPEN), (fr, -0.17, P.MARKER)):
                ax.plot(np.percentile(v, [25, 75]), [y + dy] * 2, color=col, lw=1.6, alpha=0.35, zorder=2,
                        solid_capstyle="butt")
                ax.plot(np.median(v), y + dy, ls="none", color=col, zorder=3, **{**mk, "markersize": 4})
        ax.set_title(title)
        ax.set_xlim(-0.05, 1.05)
        ax.set_xticks([0, 0.5, 1])
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
    label_rows(axes[0], order)
    handles = [plt.Line2D([], [], ls="none", color="black", label="near recalls", **OPEN),
               plt.Line2D([], [], ls="none", color="black", markerfacecolor=P.DARK, label="far recalls",
                          **P.MARKER)]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.13), ncol=2, fontsize=6.5)
    fig.supxlabel("Accuracy", fontsize=7, fontweight="bold", y=-0.04)
    save(fig, "near_far_recall")


# scratch vs SL2 -> BabyLM

PRE_ORDER = ["pure_tf", "4s4t_alternate", "6s2t_start", "pure_ssm"]


def best_lr(init, c):
    """BabyLM validation PPL per epoch (seeds x epochs) at whichever of the two learning rates gives the
    lower median best PPL. init: "none" (BabyLM only) or "sl2" (SL2 -> BabyLM)."""
    cands = [[json.load(open(p)) for p in glob.glob(os.path.join(SYN, d, init, c, "*.json"))]
             for d in ("prepretrain", "prepretrain_lrswap")]
    rs = min((r for r in cands if r), key=lambda r: np.median([x["best_ppl"] for x in r]))
    return np.array([x["val_ppls"] for x in rs])


def prepretrain():
    order = PRE_ORDER

    fig, axes = plt.subplots(1, 2, figsize=(P.ACL_COL, 1.75), sharey=True)
    fig.subplots_adjust(wspace=0.12)
    for ax, (init, title, ls) in zip(axes, [("none", "BabyLM", (0, (2, 1.5))), ("sl2", "SL2 → BabyLM", "-")]):
        for c in order:
            arr = best_lr(init, c)
            ep = np.arange(1, arr.shape[1] + 1)
            med = np.median(arr, axis=0)
            ax.fill_between(ep, np.percentile(arr, 25, axis=0), np.percentile(arr, 75, axis=0),
                            color=P.color(c), alpha=0.15, lw=0)   # IQR over seeds
            ax.plot(ep, med, color=P.color(c), lw=1.1, ls=ls)
        if init == "sl2":   # one legend, right panel
            leg = ax.legend(handles=[plt.Line2D([], [], color=P.color(c), lw=1.1, label=P.LAYOUT[c]) for c in order],
                            loc="upper right", handlelength=1.0, handletextpad=0.4, borderaxespad=0.2,
                            labelspacing=0.25, prop={"family": "DejaVu Sans Mono", "weight": "bold", "size": 4.8})
            for t, c in zip(leg.get_texts(), order):
                t.set_color(P.color(c))
        ax.set_title(title)
        ax.set_xlim(1, 20)
        ax.set_xticks([1, 10, 20])
        ax.set_xlabel("Epoch", fontsize=6.5)
        ax.tick_params(labelsize=5.8)
    axes[0].set_ylim(21, 30)
    axes[0].set_ylabel("BabyLM validation PPL", fontsize=6.5)
    save(fig, "babylm_prepretrain")


def epoch20():
    """Slope chart: BabyLM PPL at epoch 20 (median over seeds) without and with SL2 pre-pretraining, with
    Kendall's tau between the two rankings."""
    from plot_rank_slope import spread
    lo, hi = 22.0, 24.5
    f = lambda v: (v - lo) / (hi - lo)   # y in [0, 1]
    xl, xr = 0.0, 1.0
    fig, ax = plt.subplots(figsize=(P.ACL_COL, 2.55))
    fig.subplots_adjust(left=0.0, right=1.0, top=0.97, bottom=0.09)
    for x in (xl, xr):
        ax.plot([x, x], [0, 1], color=P.GUIDE, lw=1.2, zorder=1, solid_capstyle="round")
        for t in np.arange(lo, hi + 1e-9, 0.5):
            ax.plot([x - 0.025, x], [f(t)] * 2, color=P.GUIDE, lw=1.2, solid_capstyle="butt", zorder=1)
            if x == xl:
                ax.text(x - 0.04, f(t), f"{t:g}", ha="right", va="center", fontsize=6.3, color="#555555")
    ends = []
    for c in PRE_ORDER:
        a, b = (np.array([r["val_ppls"][19] for r in runs(f"prepretrain/{i}", c)]) for i in ("none", "sl2"))   # epoch 20
        ya, yb = f(np.median(a)), f(np.median(b))
        ax.plot([xl, xr], [ya, yb], color=P.color(c), lw=2.0, zorder=2, solid_capstyle="round")
        ax.plot([xl, xr], [ya, yb], ls="none", color=P.color(c), zorder=3, **P.MARKER)
        ends.append((c, np.median(a), np.median(b)))
    for (c, _, vb), y in zip(ends, spread([f(e[2]) for e in ends], 0.083)):
        ax.text(xr + 0.06, y, P.LAYOUT[c], ha="left", va="center", fontsize=6.6, family="DejaVu Sans Mono",
                fontweight="bold", color=P.color(c))
    # Kendall's tau between the two rankings
    tau = np.mean([np.sign(p[1] - q[1]) * np.sign(p[2] - q[2]) for p, q in itertools.combinations(ends, 2)])
    ax.text(0.5, -0.08, f"τ = {tau:.2f}".replace("-", "−"), ha="center", va="center", fontsize=6.6,
            fontstyle="italic", color="#555555")
    ax.set_xlim(-0.86, 1.86)   # symmetric margins
    ax.set_ylim(-0.13, 1.04)
    ax.set_xticks([xl, xr], ["BabyLM", "SL2 → BabyLM"])
    ax.set_yticks([])
    for sp in ("left", "bottom"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="x", length=0, pad=4)
    ax.text(xl - 0.40, 0.5, "BabyLM PPL at epoch 20", rotation=90, ha="center", va="center", fontsize=7.5)
    name = "babylm_prepretrain_ep20"
    fig.savefig(os.path.join(OUT, name + ".png"))   # no tight bbox, keeps the margins
    print("saved", os.path.normpath(os.path.join(OUT, name + ".png")))

# KL across language parameters

def sensitivity_kl():
    """As sensitivity.png (plot_paper_figs.py) with within-grammar and within-non-grammar KL in place of
    recall. SL2 sweep, window 20, median and IQR over five seeds."""
    cfgs = ["pure_tf", "4s4t_alternate", "6s2t_start", "7s1t_mid", "6d2t_start"]
    cols = [("Grammar words", [(5, "ng5"), (10, None), (20, "ng20"), (40, "ng40")]),
            ("Non-grammar words", [(25, "nng25"), (65, None), (150, "nng150"), (390, "nng390")]),
            ("Fact density", [(0.01, "fr01"), (0.02, "fr02"), (0.04, None), (0.08, "fr08")])]
    rows = [("kl_within_grammar", "Within-grammar KL"), ("kl_within_non_grammar", "Within-non-grammar KL")]
    fig, axes = plt.subplots(2, 3, figsize=(P.ACL_COL, 2.7), sharey="row", gridspec_kw={"wspace": 0.22,
                                                                                         "hspace": 0.45})
    for i, (field, ylab) in enumerate(rows):
        for j, (title, pts) in enumerate(cols):
            ax = axes[i, j]
            xs = [x for x, _ in pts]
            for c in cfgs:
                vals = [np.array([at_best(r, "kl_divs", field) for r in
                                  runs("lang_sl2_w20" if name is None else f"lang_sl2_sweep/{name}_w20", c)])
                        for _, name in pts]
                med = [np.median(v) for v in vals]
                ax.fill_between(xs, [np.percentile(v, 25) for v in vals], [np.percentile(v, 75) for v in vals],
                                color=P.color(c), alpha=0.18, lw=0, zorder=1)   # IQR over seeds
                ax.plot(xs, med, color=P.color(c), lw=1.4, zorder=2)
                ax.plot(xs, med, ls="none", color=P.color(c), zorder=3,
                        **{**P.MARKER, "markersize": 2.6, "markeredgewidth": 0.5})
            ax.set_xscale("log")
            ax.set_xticks(xs, [f"{x:g}" for x in xs], fontsize=5.6)
            ax.minorticks_off()
            ax.tick_params(axis="y", labelsize=5.6)
            if i == 0:
                ax.set_title(title, fontsize=6.0)
        axes[i, 0].set_ylabel(ylab, fontsize=5.8)
        axes[i, 0].set_ylim(bottom=0)
    handles = [plt.Line2D([], [], color=P.color(c), lw=1.6, label=P.LAYOUT[c]) for c in cfgs]
    fig.canvas.draw()   # centre the legend on the panels
    cx = (axes[0, 0].get_position().x0 + axes[0, 2].get_position().x1) / 2
    leg = fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(cx, 0.93), ncol=len(cfgs),
                     handlelength=0.9, handletextpad=0.3, columnspacing=0.6,
                     prop={"family": "DejaVu Sans Mono", "weight": "bold", "size": 4.6})
    for t, c in zip(leg.get_texts(), cfgs):
        t.set_color(P.color(c))
    save(fig, "sensitivity_kl")


# SL2 vs BabyLM scatter

def sl2_babylm_scatter(babylm="prepretrain/none", name="sl2_babylm_scatter"):
    """Best validation PPL on SL2 (20k x 15) against best BabyLM PPL (20 epochs), window 20, medians over
    seeds, with Kendall's tau. babylm: BabyLM results directory under results/synthetic."""
    cfgs = ["pure_tf", "4s4t_alternate", "6s2t_start", "6s2t_even", "7s1t_mid", "pure_ssm",
            "pure_dp", "4d4t_alternate", "6d2t_start"]
    LABEL_AT = {"pure_tf": (-4, 0, "right", "center")}
    pts = {}
    for c in cfgs:
        x = np.array([min(r["val_ppls"]) for r in runs("lang_sl2_w20", c)])
        y = np.array([json.load(open(p))["best_ppl"] for p in glob.glob(os.path.join(SYN, babylm, c, "*.json"))])
        pts[c] = (x, y)
    fig, ax = plt.subplots(figsize=(P.ACL_COL, 2.4))
    for c, (x, y) in pts.items():
        mx, my = np.median(x), np.median(y)
        col = P.color(c)
        ax.plot(mx, my, ls="none", color=col, zorder=3, **{**P.MARKER, "markersize": 4.5})
        dx, dy, ha, va = LABEL_AT.get(c, (4, 0, "left", "center"))
        ax.annotate(P.LAYOUT[c], (mx, my), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va,
                    fontsize=5.0, family="DejaVu Sans Mono", fontweight="bold", color=col)
    med = [(np.median(x), np.median(y)) for x, y in pts.values()]
    tau = np.mean([np.sign(a[0] - b[0]) * np.sign(a[1] - b[1]) for a, b in itertools.combinations(med, 2)])
    ax.text(0.03, 0.97, f"τ = {tau:.2f}".replace("-", "−"), transform=ax.transAxes, ha="left", va="top",
            fontsize=6.6, fontstyle="italic", color="#555555")
    ax.set_xlabel("SL2 PPL", fontsize=7)
    ax.set_ylabel("BabyLM PPL", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.set_xlim(right=ax.get_xlim()[1] + 0.35)   # room for labels
    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.5))
    save(fig, name)

# sl-eng, 100k sentences x 15 epochs

def long_training():
    """Top: Kendall's tau between the SL2 ranking and the sl-eng ranking at each epoch. Bottom: per-sentence
    recall (evaluated every third epoch), median over seeds. Dashed line: share of recalls whose value is
    within 19 tokens. Dotted line: epoch 3, the SL2 token budget."""
    d = "lang_sl2_sweep/sl-eng-d04_w20_n100k"
    cfgs = ["pure_tf", "4s4t_alternate", "6s2t_start", "6s2t_even", "7s1t_mid", "pure_ssm",
            "pure_dp", "4d4t_alternate", "6d2t_start"]
    sl2 = np.array([np.median([min(r["val_ppls"]) for r in runs("lang_sl2_w20", c)]) for c in cfgs])
    ppl = np.array([np.median([r["val_ppls"] for r in runs(d, c)], axis=0) for c in cfgs])   # cfg x epoch
    tau = [np.mean([np.sign(sl2[i] - sl2[j]) * np.sign(ppl[i, e] - ppl[j, e])
                    for i, j in itertools.combinations(range(len(cfgs)), 2)]) for e in range(ppl.shape[1])]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(P.ACL_COL, 3.1), sharex=True,
                                   gridspec_kw={"height_ratios": [1, 1.6], "hspace": 0.12})
    ep = np.arange(1, len(tau) + 1)
    ax1.plot(ep, tau, color=P.DARK, lw=1.2, zorder=2)
    ax1.plot(ep, tau, ls="none", color=P.DARK, zorder=3, **{**P.MARKER, "markersize": 3.5})
    ax1.set_ylim(0, 0.85)
    ax1.set_ylabel("τ with SL2", fontsize=6.5)

    for c in cfgs[1:] + cfgs[:1]:   # pure attention drawn last
        rs = runs(d, c)
        acc = np.array([[e["accuracy_sent"] for e in r["recall_accs"]] for r in rs])
        x = np.array([e["epoch"] for e in rs[0]["recall_accs"]]) + 1   # stored 0-based
        ax2.plot(x, np.median(acc, axis=0), color=P.color(c), lw=1.1, zorder=2)
        ax2.plot(x, np.median(acc, axis=0), ls="none", color=P.color(c), zorder=3, **{**P.MARKER, "markersize": 3.5})
    near = json.load(open(os.path.join(SYN, d, "recall_by_distance.json")))["frac_near"]
    ax2.hlines(near, 0.5, 15, color=P.GUIDE, ls="--", lw=1, zorder=0)
    ax2.text(0.6, near + 0.03, "inside the\nattention window", ha="left", va="bottom", fontsize=5.3, color="#555555",
             linespacing=1.0)
    ax2.set_ylim(-0.04, 1.06)
    ax2.set_ylabel("Recall, per sentence", fontsize=6.5)
    ax2.set_xlabel("Epoch on sl-eng (100k sentences)", fontsize=6.5)
    ax2.set_xlim(0.5, 15.5)
    ax2.set_xticks([1, 3, 6, 9, 12, 15])
    for ax in (ax1, ax2):
        ax.axvline(3, color=P.GUIDE, ls=":", lw=1, zorder=0)
        ax.tick_params(labelsize=5.8)
    ax1.text(3.25, 0.06, "SL2 token budget", ha="left", va="bottom", fontsize=5.3, color="#555555")

    leg = fig.legend(handles=[plt.Line2D([], [], color=P.color(c), lw=1.1, label=P.LAYOUT[c]) for c in cfgs],
                     loc="upper center", bbox_to_anchor=(0.53, -0.005), ncol=5, handlelength=1.0,
                     handletextpad=0.4, columnspacing=0.9, labelspacing=0.25,
                     prop={"family": "DejaVu Sans Mono", "weight": "bold", "size": 4.5})
    for t, c in zip(leg.get_texts(), cfgs):
        t.set_color(P.color(c))
    save(fig, "sleng_long_training")


if __name__ == "__main__":
    style()
    for name in sys.argv[1:] or ["orderings", "nearfar"]:
        {"orderings": orderings, "nearfar": nearfar, "prepretrain": prepretrain, "epoch20": epoch20,
         "sensitivity_kl": sensitivity_kl, "sl2_babylm_scatter": sl2_babylm_scatter, "long_training": long_training}[name]()
