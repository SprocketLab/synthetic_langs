#!/usr/bin/env python3
"""Paper figures, one ACL column each (style in paper_style.py).

  window_ablation  appendix: recall per architecture, window 20 vs full attention, SL2 and sl-eng
  sensitivity      Sec. 5.3: SL2 recall as one language parameter varies
  budget           Sec. 6.2: BabyLM PPL vs length of SL2 pre-pretraining
  pipelines        Sec. 6.3: MKAR / MQAR accuracy after each pre-training pipeline
  convergence      appendix: BabyLM validation PPL over 20 and 40 epochs
  tasks_rank       Sec. 3: rank of each architecture on MKAR, MQAR and BabyLM

Medians over seeds at the epoch with the lowest validation PPL (recall at the nearest evaluation);
bands are interquartile ranges.

    python plot_paper_figs.py [window_ablation sensitivity budget ...]   # -> figures/
"""
import os, sys, json, glob, itertools
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import paper_style as P

_HERE = os.path.dirname(os.path.abspath(__file__))
SYN = os.path.join(_HERE, "..", "..", "results", "synthetic")
OUT = os.path.join(_HERE, "..", "..", "figures")
os.makedirs(OUT, exist_ok=True)


def runs(d, cfg):
    return [json.load(open(p)) for p in sorted(glob.glob(os.path.join(SYN, d, cfg, "*seed*.json")))]


def at_best(r, key, field):
    b = int(np.argmin(r["val_ppls"]))
    return min(r[key], key=lambda e: (abs(e["epoch"] - b), e["epoch"]))[field]


def best_ppl(rs):
    return [min(r["val_ppls"]) for r in rs]


def recall(rs):
    return [at_best(r, "recall_accs", "accuracy_sent") for r in rs]


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("saved", os.path.normpath(os.path.join(OUT, name + ".{pdf,png}")))


def panel_title(ax, text):
    ax.set_title(text, fontsize=6.5, fontstyle="italic", pad=3)


# window 20 vs full attention

def window_ablation():
    cfgs = ["pure_tf", "4s4t_alternate", "6s2t_start", "6s2t_even", "7s1t_mid", "4d4t_alternate", "6d2t_start"]
    panels = [("SL2", "lang_sl2_w20", "lang_sl2_w200"),
              ("sl-eng", "lang_sl2_sweep/sl-eng-d04_w20", "lang_sl2_sweep/sl-eng-d04_w200")]
    fig, axes = plt.subplots(1, 2, figsize=(P.ACL_COL, 2.2), sharey=True, gridspec_kw={"wspace": 0.12})
    ys = np.arange(len(cfgs))[::-1]
    for ax, (title, d20, dfull) in zip(axes, panels):
        ceiling = runs(d20, cfgs[0])[0]["recall_ceiling_sent"]
        ax.axvline(ceiling, color=P.GUIDE, ls="--", lw=0.9, zorder=0)
        for c, y in zip(cfgs, ys):
            a, b = np.median(recall(runs(d20, c))), np.median(recall(runs(dfull, c)))
            ax.plot([a, b], [y, y], color=P.color(c), lw=2.0, solid_capstyle="round", zorder=1)
            # open marker: window 20; filled square on top: full attention
            ax.plot([a], [y], ls="none", marker="s", markersize=7.2, markerfacecolor="white",
                    markeredgecolor=P.color(c), markeredgewidth=1.6, zorder=2)
            ax.plot([b], [y], ls="none", color=P.color(c), zorder=3, **{**P.MARKER, "markersize": 4.6})
        panel_title(ax, title)
        ax.set_xlim(-0.05, 1.05)
        ax.set_xticks([0, 0.5, 1], ["0", "0.5", "1"])
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
    fig.supxlabel("Recall accuracy", fontsize=7, fontweight="bold", y=-0.06, x=0.6)
    axes[0].set_yticks(ys, [P.LAYOUT[c] for c in cfgs], family="DejaVu Sans Mono", fontsize=6.3)
    for t, c in zip(axes[0].get_yticklabels(), cfgs):
        t.set_color(P.color(c))
    # marker key
    h_open = plt.Line2D([], [], ls="none", marker="s", markersize=5, markerfacecolor="white",
                        markeredgecolor="black", markeredgewidth=1.4, label="window 20")
    h_full = plt.Line2D([], [], ls="none", color="#888888", label="full attention",
                        **{**P.MARKER, "markersize": 4.6})
    h_ceil = plt.Line2D([], [], color=P.GUIDE, ls="--", lw=0.9, label="ceiling")
    fig.legend(handles=[h_open, h_full, h_ceil], loc="lower center", bbox_to_anchor=(0.55, 0.98), ncol=3,
               fontsize=6.3, handletextpad=0.3, columnspacing=1.0, handlelength=1.4)
    save(fig, "window_ablation")


# sensitivity to SL2 language parameters

def sensitivity():
    cfgs = ["pure_tf", "4s4t_alternate", "6s2t_start", "7s1t_mid", "6d2t_start"]
    axes_def = [("Grammar", [(5, "ng5"), (10, None), (20, "ng20"), (40, "ng40")], "log"),
                ("Non-grammar", [(25, "nng25"), (65, None), (150, "nng150"), (390, "nng390")], "log"),
                ("Fact density", [(0.01, "fr01"), (0.02, "fr02"), (0.04, None), (0.08, "fr08")], "log")]
    fig, axes = plt.subplots(1, 3, figsize=(P.ACL_COL, 1.6), sharey=True, gridspec_kw={"wspace": 0.22})
    for ax, (title, pts, scale) in zip(axes, axes_def):
        xs = [x for x, _ in pts]
        for c in cfgs:
            med, lo, hi = [], [], []
            for x, name in pts:
                v = recall(runs("lang_sl2_w20" if name is None else f"lang_sl2_sweep/{name}_w20", c))
                med.append(np.median(v)); lo.append(np.percentile(v, 25)); hi.append(np.percentile(v, 75))
            ax.fill_between(xs, lo, hi, color=P.color(c), alpha=0.18, lw=0, zorder=1)
            ax.plot(xs, med, color=P.color(c), lw=1.4, zorder=2)
            ax.plot(xs, med, ls="none", color=P.color(c), zorder=3,
                    **{**P.MARKER, "markersize": 2.6, "markeredgewidth": 0.5})
        ax.set_xscale(scale)
        ax.set_xticks(xs, [f"{x:g}" for x in xs], fontsize=5.6)
        ax.minorticks_off()
        panel_title(ax, title)
    axes[0].set_ylabel("Recall accuracy", fontsize=6.3)
    axes[0].set_ylim(-0.03, 1.0)
    axes[0].tick_params(axis="y", labelsize=5.6)
    handles = [plt.Line2D([], [], color=P.color(c), lw=1.6, label=P.LAYOUT[c]) for c in cfgs]
    leg = fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.52, 0.98), ncol=3,
                     handlelength=1.2, handletextpad=0.4, columnspacing=0.8, labelspacing=0.3,
                     prop={"family": "DejaVu Sans Mono", "weight": "bold", "size": 5.4})
    for t, c in zip(leg.get_texts(), cfgs):
        t.set_color(P.color(c))
    save(fig, "sensitivity")


# length of SL2 pre-pretraining

def budget():
    cfgs = ["pure_tf", "4s4t_alternate", "6s2t_start", "pure_ssm"]
    eps = ["0", "0.1", "0.3", "1", "3", "15"]
    fig, ax = plt.subplots(figsize=(P.ACL_COL, 1.9))
    xs = np.arange(len(eps))
    for c in cfgs:
        med, lo, hi = [], [], []
        for e in eps:
            v = [json.load(open(p))["best_ppl"] for p in glob.glob(os.path.join(SYN, "prepretrain_budget", f"ep{e}", c, "*.json"))]
            med.append(np.median(v)); lo.append(np.percentile(v, 25)); hi.append(np.percentile(v, 75))
        ax.fill_between(xs, lo, hi, color=P.color(c), alpha=0.18, lw=0, zorder=1)
        ax.plot(xs, med, color=P.color(c), lw=1.5, zorder=2)
        ax.plot(xs, med, ls="none", color=P.color(c), zorder=3, **{**P.MARKER, "markersize": 2.8, "markeredgewidth": 0.5})
        ax.text(xs[-1] + 0.15, med[-1], P.LAYOUT[c], ha="left", va="center", fontsize=6.3,
                family="DejaVu Sans Mono", fontweight="bold", color=P.color(c))
    ax.set_xticks(xs, eps)
    ax.set_xlim(-0.2, len(eps) + 0.9)
    ax.set_xlabel("SL2 pre-pretraining epochs", fontsize=7)
    ax.set_ylabel("BabyLM PPL", fontsize=7)
    save(fig, "budget")


# MKAR / MQAR after each pre-training pipeline

def pipelines():
    cfgs = ["pure_tf", "4s4t_alternate", "6s2t_start", "pure_ssm"]
    stages = [("Scratch", "posttrain_sl2_v2/{t}/scratch_matched/{c}"),
              ("SL2", "posttrain_sl2_v2/{t}/full/{c}"),
              ("BabyLM", "prepretrain_post/none/{t}/full/{c}"),
              ("SL2→\nBabyLM", "prepretrain_post/sl2/{t}/full/{c}")]
    fig, axes = plt.subplots(1, 2, figsize=(P.ACL_COL, 1.75), sharey=True, gridspec_kw={"wspace": 0.12})
    xs = np.arange(len(stages))
    for ax, (task, title) in zip(axes, [("mkar", "MKAR"), ("mqar", "MQAR")]):
        for k, c in enumerate(cfgs):
            vals = [[json.load(open(p))["mean_acc"] for p in glob.glob(os.path.join(SYN, d.format(t=task, c=c), "*.json"))]
                    for _, d in stages]
            off = (k - 1.5) * 0.07
            med = [np.median(v) for v in vals]
            lo = [med[i] - np.percentile(v, 25) for i, v in enumerate(vals)]
            hi = [np.percentile(v, 75) - med[i] for i, v in enumerate(vals)]
            eb = ax.errorbar(xs + off, med, yerr=[lo, hi], fmt="none", ecolor=P.color(c), elinewidth=0.7, capsize=0,
                             zorder=1)   # IQR over seeds
            for art in eb.lines[2]:
                art.set_alpha(0.35)
            ax.plot(xs + off, med, color=P.color(c), lw=1.4, zorder=2)
            ax.plot(xs + off, med, ls="none", color=P.color(c), zorder=3,
                    **{**P.MARKER, "markersize": 2.8, "markeredgewidth": 0.5})
        ax.set_xticks(xs, [n for n, _ in stages], fontsize=5.4)
        ax.set_xlim(-0.35, len(stages) - 0.65)
        panel_title(ax, title)
    axes[0].set_ylim(0, 1.03)
    axes[0].set_ylabel("Accuracy", fontsize=6.3)
    axes[0].tick_params(axis="y", labelsize=5.6)
    handles = [plt.Line2D([], [], color=P.color(c), lw=1.6, label=P.LAYOUT[c]) for c in cfgs]
    leg = fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.52, 0.98), ncol=4,
                     handlelength=1.2, handletextpad=0.4, columnspacing=0.8,
                     prop={"family": "DejaVu Sans Mono", "weight": "bold", "size": 5.4})
    for t, c in zip(leg.get_texts(), cfgs):
        t.set_color(P.color(c))
    save(fig, "pipelines")


# BabyLM training length

def convergence():
    cfgs = ["pure_tf", "4s4t_alternate", "6s2t_start", "4d4t_alternate"]
    from plot_rank_slope import spread
    fig, ax = plt.subplots(figsize=(P.ACL_COL, 1.9))
    ends = []
    for c in cfgs:
        for d, ls in (("none", (0, (2, 1.5))), ("none_ep40", "-")):
            rs = [json.load(open(p)) for p in glob.glob(os.path.join(SYN, "prepretrain", d, c, "*.json"))]
            arr = np.array([r["val_ppls"] for r in rs])
            v = np.median(arr, axis=0)
            ep = np.arange(1, len(v) + 1)
            ax.fill_between(ep, np.percentile(arr, 25, axis=0), np.percentile(arr, 75, axis=0),
                            color=P.color(c), alpha=0.15, lw=0)   # IQR over seeds
            ax.plot(ep, v, color=P.color(c), lw=1.3, ls=ls)
        ends.append(v[-1])
    for c, y in zip(cfgs, spread(ends, 0.9)):
        ax.text(41, y, P.LAYOUT[c], ha="left", va="center", fontsize=5.8,
                family="DejaVu Sans Mono", fontweight="bold", color=P.color(c))
    ax.set_xlim(0, 52)
    ax.set_ylim(19.5, 34)
    ax.set_xticks([1, 10, 20, 30, 40])
    ax.set_xlabel("Epoch", fontsize=7)
    ax.set_ylabel("BabyLM validation PPL", fontsize=7)
    ax.tick_params(labelsize=6)
    h = [plt.Line2D([], [], color="#777777", lw=1.3, ls=(0, (2, 1.5)), label="20 epochs"),
         plt.Line2D([], [], color="#777777", lw=1.3, label="40 epochs")]
    ax.legend(handles=h, loc="upper right", fontsize=6, handlelength=1.8)
    save(fig, "convergence")


# synthetic tasks vs BabyLM

def tasks_rank():
    cfgs = ["pure_tf", "4s4t_alternate", "6s2t_start", "6s2t_even", "7s1t_mid", "pure_ssm"]
    med = lambda pat, key: np.median([json.load(open(p))[key] for p in glob.glob(os.path.join(SYN, pat, "*.json"))])
    # BabyLM in the middle column
    cols = [("MKAR", {c: -med(f"mkar_d256_full/d256/{c}", "mean_acc") for c in cfgs}),
            ("BabyLM", {c: med(f"prepretrain/none/{c}", "best_ppl") for c in cfgs}),
            ("MQAR", {c: -med(f"mqar_d256_full/d256/{c}", "mean_acc") for c in cfgs})]
    rank = {name: {c: r + 1 for r, c in enumerate(sorted(cfgs, key=v.get))} for name, v in cols}
    fig, ax = plt.subplots(figsize=(P.ACL_COL, 1.9))
    fig.subplots_adjust(left=0.13, right=0.78, top=0.93, bottom=0.13)
    xs = np.arange(len(cols))
    for x in xs:
        ax.plot([x, x], [0.7, len(cfgs) + 0.3], color=P.GUIDE, lw=1.0, zorder=0)
    for c in cfgs:
        ys = [rank[n][c] for n, _ in cols]
        ax.plot(xs, ys, color=P.color(c), lw=1.8, zorder=2)
        ax.plot(xs, ys, ls="none", color=P.color(c), zorder=3, **{**P.MARKER, "markersize": 4.2, "markeredgewidth": 0.7})
        ax.text(xs[-1] + 0.12, ys[-1], P.LAYOUT[c], ha="left", va="center", fontsize=6.0,
                family="DejaVu Sans Mono", fontweight="bold", color=P.color(c))
    # Kendall's tau between each task and BabyLM
    for i in (0, 1):
        a, b = rank[cols[i][0]], rank[cols[i + 1][0]]
        tau = np.mean([np.sign(a[p] - a[q]) * np.sign(b[p] - b[q]) for p, q in itertools.combinations(cfgs, 2)])
        ax.text(i + 0.5, len(cfgs) + 0.75, f"τ = {tau:.2f}".replace("-", "−"), ha="center", va="center", fontsize=6.3,
                fontstyle="italic", color="#555555")
    ax.set_xticks(xs, [n for n, _ in cols], fontsize=7)
    ax.set_xlim(-0.15, len(cols) - 0.85)
    ax.set_ylim(len(cfgs) + 1.0, 0.6)   # rank 1 at the top
    ax.set_yticks(range(1, len(cfgs) + 1))
    ax.set_ylabel("Rank (1 = best)", fontsize=7)
    ax.tick_params(axis="y", labelsize=6, length=0)
    ax.tick_params(axis="x", length=0)
    for s_ in ("left", "bottom"):
        ax.spines[s_].set_visible(False)
    os.makedirs(OUT, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"tasks_babylm.{ext}"))
    plt.close(fig)
    print("saved", os.path.normpath(os.path.join(OUT, "tasks_babylm.{pdf,png}")))


if __name__ == "__main__":
    P.apply()
    todo = sys.argv[1:] or ["window_ablation", "sensitivity", "budget", "pipelines", "convergence", "tasks_rank"]
    for name in todo:
        globals()[name]()
