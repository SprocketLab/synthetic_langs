#!/usr/bin/env python3
"""Decode-recall sweep at d=256 over every configuration in decode_recall.CONFIGS, 5 seeds.

Results -> results/synthetic/d256/ (finished runs are skipped on restart).
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import os, sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import decode_recall as RE

RE.DEFAULTS["hidden_size"] = 256

SEEDS = [0, 1, 2, 3, 4]
ALL_CONFIGS = list(RE.CONFIGS.keys())

_BASE = os.path.abspath(os.path.join(_HERE, "..", "..", "results", "synthetic"))
RE.RESULTS_DIR = _BASE

print(f"\n{'='*60}", flush=True)
print(f"=== Decode-Recall d=256 full sweep - {len(ALL_CONFIGS)} configs x {len(SEEDS)} seeds ===", flush=True)
print(f"{'='*60}", flush=True)

print(f"\nStep 1: auto-LR search per config", flush=True)
lrs = {c: RE.find_best_lr(c) for c in ALL_CONFIGS}
print(f"\nResolved auto-LR:", flush=True)
for c in ALL_CONFIGS:
    print(f"  {c:20s} {RE.CONFIGS[c]}  lr={lrs[c]:.2e}", flush=True)

plan = [(c, lrs[c], s) for c in ALL_CONFIGS for s in SEEDS]
print(f"\nStep 2: {len(plan)} training runs", flush=True)
RE.run_plan(plan, "decode_recall_d256_full")

print("=== Decode-Recall d=256 done ===", flush=True)
