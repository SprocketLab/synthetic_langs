#!/usr/bin/env python3
"""SL2 language-parameter sweep (Section 5.3).

One parameter at a time is varied around the base language langs/sl2 (SL2_Config defaults:
N_GRAMMAR=10, 65 non-grammar words, FACT_DENSITY=RECALL_DENSITY=0.04, OTHER_DENSITY=0.2):

  n_grammar     number of grammar words           (non-grammar count held at 65)
  n_non_grammar number of non-grammar words       (N_VOCAB = N_GRAMMAR + n)
  fact_density  fact / recall density (kept equal) (OTHER_DENSITY held at 0.2)

Each point is a language generated from SL2_Config defaults + overrides with LANG_SEED by
`synthetic_lang_sl2.py --lang <name>`, saved in results/synthetic/lang_sl2_sweep/<name>/language/.

Usage
  python sl2_sweep.py list             table of sweep points
  python sl2_sweep.py jobs             one command line per language
  python synthetic_lang_sl2.py --lang ng5 [--configs ... --seeds ...]      run one point
"""
import sys

BASE_NON_GRAMMAR = 65          # SL2_Config: N_VOCAB 75 - N_GRAMMAR 10
BASE = dict(N_GRAMMAR=10, N_VOCAB=75, FACT_DENSITY=0.04, RECALL_DENSITY=0.04, OTHER_DENSITY=0.2)

# name -> (axis, x-value, SL2_Config overrides on top of the defaults); names become directory names
SWEEP = {
    "base":   (None, None, {}),
    # axis 1: number of grammar words (non-grammar count fixed at 65)
    "ng5":    ("n_grammar", 5,  dict(N_GRAMMAR=5,  N_VOCAB=5 + BASE_NON_GRAMMAR)),
    "ng20":   ("n_grammar", 20, dict(N_GRAMMAR=20, N_VOCAB=20 + BASE_NON_GRAMMAR)),
    "ng40":   ("n_grammar", 40, dict(N_GRAMMAR=40, N_VOCAB=40 + BASE_NON_GRAMMAR)),
    # axis 2: number of non-grammar words (grammar fixed at 10)
    "nng25":  ("n_non_grammar", 25,  dict(N_VOCAB=10 + 25)),
    "nng150": ("n_non_grammar", 150, dict(N_VOCAB=10 + 150)),
    "nng390": ("n_non_grammar", 390, dict(N_VOCAB=10 + 390)),
    # axis 3: fact / recall density (equal), other-word density fixed at 0.2
    "fr01":   ("fact_density", 0.01, dict(FACT_DENSITY=0.01, RECALL_DENSITY=0.01)),
    "fr02":   ("fact_density", 0.02, dict(FACT_DENSITY=0.02, RECALL_DENSITY=0.02)),
    "fr08":   ("fact_density", 0.08, dict(FACT_DENSITY=0.08, RECALL_DENSITY=0.08)),
}

# `base` reuses the saved langs/sl2 instead of drawing a new language
COPY_FROM = {"base": "sl2"}

AXES = {
    "n_grammar":     ("# grammar words", 10),
    "n_non_grammar": ("# non-grammar words", 65),
    "fact_density":  ("fact = recall density", 0.04),
}

LANG_SEED = 0          # seed for random / np.random when generating a language


def points_on_axis(axis):
    """[(x, name)] for one axis, base included, sorted by x."""
    pts = [(AXES[axis][1], "base")]
    pts += [(x, n) for n, (a, x, _) in SWEEP.items() if a == axis]
    return sorted(pts)


def overrides(name):
    if name not in SWEEP:
        raise KeyError(f"unknown sweep point {name!r}; known: {', '.join(SWEEP)}")
    return dict(SWEEP[name][2])


def describe(name):
    cfg = dict(BASE); cfg.update(overrides(name))
    return (f"N_GRAMMAR={cfg['N_GRAMMAR']} non_grammar={cfg['N_VOCAB'] - cfg['N_GRAMMAR']} "
            f"fact={cfg['FACT_DENSITY']} recall={cfg['RECALL_DENSITY']} other={cfg['OTHER_DENSITY']}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "list"
    if cmd == "list":
        for name, (axis, x, ov) in SWEEP.items():
            print(f"{name:8s} {str(axis):14s} {str(x):5s} {describe(name)}")
    elif cmd == "jobs":
        for name in SWEEP:
            print(f"synth_lang/synthetic_lang_sl2.py sl2_sweep_{name} --lang {name}")
    else:
        sys.exit(__doc__)
