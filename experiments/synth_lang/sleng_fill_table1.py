#!/usr/bin/env python3
"""sl-eng (langs/sl-eng-d04) runs for Table 1 and the appendix table, one configuration:

  1. 20k sentences x 15 epochs, window 20, auto-LR over 1e-4..3e-3 (10 points), no early stopping
     -> lang_sl2_sweep/sl-eng-d04_w20 (appendix table)
  2. 100k sentences x 3 epochs (same number of tokens as SL2) at the learning rate from step 1
     -> lang_sl2_sweep/sl-eng-d04_w20_n100k_ep3 (Table 1)

    python sleng_fill_table1.py 4s4t_tf_first [extra args passed to both steps, e.g. --seeds 0 --epochs 1]
"""
import os, sys, json, subprocess

_HERE = os.path.dirname(os.path.abspath(__file__))
SWEEP = os.path.join(_HERE, "..", "..", "results", "synthetic", "lang_sl2_sweep")
COMMON = ["--lang-dir", "sl-eng-d04", "--window", "20", "--patience", "0"]


def run(args):
    print("+ python synthetic_lang_sl2.py " + " ".join(args), flush=True)
    subprocess.run([sys.executable, os.path.join(_HERE, "synthetic_lang_sl2.py"), *args], check=True)


def main():
    cfg, extra = sys.argv[1], sys.argv[2:]
    d20k = os.path.join(SWEEP, "sl-eng-d04_w20")
    run([*COMMON, "--lr-max", "3e-3", "--lr-points", "10", "--results-dir", d20k, "--configs", cfg, *extra])
    lr = json.load(open(os.path.join(d20k, "auto_lr.json")))[cfg]
    run([*COMMON, "--n-train", "100000", "--epochs", "3", "--results-dir", os.path.join(SWEEP, "sl-eng-d04_w20_n100k_ep3"),
         "--configs", cfg, "--lr", f"{cfg}={lr}", *extra])


if __name__ == "__main__":
    main()
