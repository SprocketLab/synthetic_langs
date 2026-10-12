#!/usr/bin/env python3
"""Stage 3 of pre-pretraining: post-train a BabyLM-trained backbone on MKAR / MQAR.

The stage-2 backbone is copied into a fresh task model as in posttrain_sl2_v2.py (variant "full":
all backbone parameters trainable, wd=0.01, LR grid 1e-5..3e-4) and the task embeddings are
re-initialised. Per-layer drift from the stage-2 weights is recorded.

Sources   from configs/stage2_manifest.json, which lists the stage-2 run and seed checkpoints per config:
            sl2   SL2 -> BabyLM -> task
            none  BabyLM -> task
Results   results/synthetic/prepretrain_post/<source>/<task>/full/<config>/<config>__seed<k>.json
          and auto_lr.json, in the posttrain_sl2_v2 format.
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import os, sys, gc, json, time, argparse
import numpy as np
import torch
from transformers import AutoTokenizer

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))

import decode_recall as RE
from model_utils import get_model
from test_utils import evaluation
from posttrain_sl2 import set_all_seeds
from lora import effective_state_dict, count_params
import posttrain_sl2_v2 as V2
import pretrain_babylm as P

MANIFEST = os.path.join(_REPO_ROOT, "configs", "stage2_manifest.json")
RESULTS_ROOT = os.path.join(_REPO_ROOT, "results", "synthetic", "prepretrain_post")
VARIANT = "full"


def load_stage2(source, config_name, seed, manifest):
    entry = manifest[source][config_name]
    ckpt = os.path.join(_REPO_ROOT, entry["checkpoints"][seed])
    if not os.path.exists(ckpt):
        raise SystemExit(f"stage-2 checkpoint missing: {ckpt}")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = get_model(P.babylm_args(config_name, entry["lr"], P.EPOCHS), tokenizer)
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    print(f"  stage-2 checkpoint loaded: {os.path.relpath(ckpt, _REPO_ROOT)} "
          f"(BabyLM best PPL {entry['mean_best_ppl']}, lr {entry['lr']:.1e})", flush=True)
    return model.to("cuda").eval(), entry


def stage3_single(config_name, seed, task, source, lr, manifest):
    rpath = os.path.join(V2.RESULTS_DIR, task, VARIANT, config_name, f"{config_name}__seed{seed}.json")
    if os.path.exists(rpath):
        print(f"  SKIP {source}/{task} {config_name} seed={seed} (exists)", flush=True)
        return
    set_all_seeds(seed)
    pre_model, entry = load_stage2(source, config_name, seed, manifest)

    print(f"\n[{source}->babylm->{task} {config_name} seed={seed}] lr={lr:.2e}", flush=True)
    set_all_seeds(seed)
    args, tokenizer, train_dataset, model, pre_eff = V2.build_model(config_name, lr, task, VARIANT, pre_model)
    n_total, n_train = count_params(model), count_params(model, True)
    model.eval()
    _, _, acc0 = evaluation(args, model.to("cuda"), tokenizer, do_print=False)
    acc_before = float(np.mean(acc0))
    print(f"    accuracy before post-training: {acc_before:.4f}", flush=True)

    t0 = time.time()
    hist, model = V2.run_epochs(args, model, tokenizer, train_dataset, VARIANT, args.epochs,
                                pre_eff=pre_eff, probe=None, log_every=8)
    model.eval()
    str_mean, _, char_acc = evaluation(args, model, tokenizer, do_print=False)
    wall = time.time() - t0

    ck = os.path.join(V2.CKPT_DIR, task, VARIANT, config_name, f"seed{seed}.pt")
    os.makedirs(os.path.dirname(ck), exist_ok=True)
    torch.save({k: v.cpu() for k, v in effective_state_dict(model, include_buffers=True).items()}, ck)

    result = {
        "config": config_name, "architecture": RE.CONFIGS[config_name], "seed": seed,
        "task": task, "variant": VARIANT,
        "pretrain_source": f"{source}->babylm", "stage2_root": entry["root"],
        "stage2_lr": entry["lr"], "stage2_best_ppl": entry["mean_best_ppl"],
        "pretrain_lr": entry["lr"], "posttrain_lr": lr, "weight_decay": V2.POSTTRAIN_WD,
        "lora": None, "vocab": "task", "pretrain_corpus": f"{source}+babylm",
        "acc_before_posttrain": acc_before,
        "posttrain_epochs": args.epochs, "hidden": P.HIDDEN_SIZE, "window": args.window,
        "params_total": n_total, "params_trainable": n_train,
        "wall_time_s": wall,
        "train_accs": hist["train_accs"], "train_losses": hist["train_losses"],
        "drift": hist["drift"], "probe": [],
        "final_acc": [float(a) for a in char_acc], "mean_acc": float(np.mean(char_acc)),
        "str_acc": [float(s) for s in str_mean], "final_loss": hist["train_losses"][-1],
        "checkpoint": os.path.relpath(ck, V2.RESULTS_DIR),
    }
    os.makedirs(os.path.dirname(rpath), exist_ok=True)
    json.dump(result, open(rpath, "w"), indent=2)
    print(f"  -> mean_acc={result['mean_acc']:.4f} best_acc={max(hist['train_accs']):.4f} ({wall:.0f}s) "
          f"final_drift={hist['drift'][-1]['backbone']:.4f}", flush=True)
    del model, tokenizer, train_dataset, pre_model
    gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", default=["sl2", "none"])
    ap.add_argument("--tasks", nargs="+", default=["mkar", "mqar"])
    ap.add_argument("--configs", nargs="+", default=["pure_tf", "4s4t_alternate", "6s2t_start", "pure_ssm"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=0, help="override post-train epochs (smoke tests)")
    ap.add_argument("--lr", type=float, default=None, help="fixed LR (skips the search)")
    ap.add_argument("--manifest", default=MANIFEST)
    ap.add_argument("--results-dir", default=None)
    a = ap.parse_args()
    if a.epochs:
        V2.V1.POSTTRAIN_EPOCHS = a.epochs
    if a.results_dir:
        RESULTS_ROOT = os.path.abspath(a.results_dir)
    manifest = json.load(open(a.manifest))

    print("=== Pre-pre-training stage 3: BabyLM checkpoint -> " + "/".join(a.tasks) + " ===", flush=True)
    print(f"sources={a.sources} configs={a.configs} seeds={a.seeds} variant={VARIANT} "
          f"wd={V2.POSTTRAIN_WD} LR grid {V2.LR_GRIDS[VARIANT][0]:.0e}..{V2.LR_GRIDS[VARIANT][-1]:.0e}", flush=True)

    for source in a.sources:
        V2.RESULTS_DIR = os.path.join(RESULTS_ROOT, source)
        V2.CKPT_DIR = os.path.join(V2.RESULTS_DIR, "checkpoints")
        for task in a.tasks:
            print(f"\n{'=' * 60}\n=== {source} -> babylm -> {task} ===\n{'=' * 60}", flush=True)
            for cfg in a.configs:
                if a.lr:
                    lr = a.lr
                else:
                    set_all_seeds(0)
                    pre0, _ = load_stage2(source, cfg, a.seeds[0], manifest)
                    lr = V2.find_best_lr(cfg, task, VARIANT, pre0)
                    del pre0; gc.collect(); torch.cuda.empty_cache()
                for seed in a.seeds:
                    stage3_single(cfg, seed, task, source, lr, manifest)
    print("\n=== stage 3 done ===", flush=True)
