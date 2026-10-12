#!/usr/bin/env python3
"""MKAR and MQAR sweep at d=256 over every configuration in decode_recall.CONFIGS, 5 seeds.

Results -> results/synthetic/mkar_d256_full/d256/ and results/synthetic/mqar_d256_full/d256/
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import decode_recall as RE

RE.DEFAULTS["hidden_size"] = 256

SEEDS = [0, 1, 2, 3, 4]
ALL_CONFIGS = list(RE.CONFIGS.keys())

_BASE = os.path.abspath(os.path.join(_HERE, "..", "..", "results", "synthetic"))

for task_name, task_label in [("mkar", "MKAR"), ("mqar", "MQAR")]:
    RE.TASK = task_name
    RE.RESULTS_DIR = os.path.join(_BASE, f"{task_name}_d256_full")

    print(f"\n{'='*60}", flush=True)
    print(f"=== {task_label} d=256 full sweep - {len(ALL_CONFIGS)} configs x {len(SEEDS)} seeds ===", flush=True)
    print(f"{'='*60}", flush=True)

    print(f"\nStep 1: auto-LR search per config on {task_label}", flush=True)
    lrs = {c: RE.find_best_lr(c) for c in ALL_CONFIGS}
    print(f"\nResolved {task_label} auto-LR:", flush=True)
    for c in ALL_CONFIGS:
        print(f"  {c:20s} {RE.CONFIGS[c]}  lr={lrs[c]:.2e}", flush=True)

    plan = [(c, lrs[c], s) for c in ALL_CONFIGS for s in SEEDS]
    print(f"\nStep 2: {len(plan)} training runs", flush=True)
    RE.run_plan(plan, f"{task_name}_d256_full")

    print(f"=== {task_label} d=256 done ===\n", flush=True)

print("=== All done ===")
