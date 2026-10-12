#!/usr/bin/env python3
"""Stage 2 of pre-pretraining: copy a synthetic-language backbone into a fresh BabyLM model and train it.

Pipeline
  1. train on a synthetic language    posttrain_sl2.pretrain_model (run here if the checkpoint is missing)
  2. train on BabyLM                  this script
  3. post-train on MKAR / MQAR        posttrain_stage3.py

The backbone (hybrid.layers.*, final_layer_norm) is copied by name and shape; embed_in / embed_out are
re-initialised for the GPT-2 vocabulary. `--sources none` trains from scratch.

Sources   sl2 (langs/sl2), any other langs/<source> (stage-1 checkpoints in
          results/synthetic/posttrain_sl2/pretrained_<source>/), or none.
Results   results/synthetic/prepretrain/<source>/<config>/<config>__seed<k>.json, auto_lr.json,
          checkpoints/<config>/seed<k>.pt

BabyLM settings: window 20, sequence length 128, batch 32, 20 epochs, wd 0.1, GPT-2 tokenizer,
500k validation tokens. d / heads / state_dim follow the stage-1 model.
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import os, sys, gc, json, math, argparse
import numpy as np
import torch
from argparse import Namespace
from torch.optim import AdamW
from accelerate import Accelerator
from transformers import AutoTokenizer

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
_LANGS_DIR = os.path.join(_REPO_ROOT, "langs")

import decode_recall as RE
from model_utils import get_model
from train_utils import ce_loss, custom_get_scheduler, train_epoch
from babylm_data import load_babylm_tokens, TextDataset
import posttrain_sl2 as V1
from posttrain_sl2 import set_all_seeds, WordTokenizer, generate_corpus, transfer_backbone

sys.path.insert(0, _LANGS_DIR)
from sl2 import SL2

# hyperparameters
HIDDEN_SIZE = V1.PRETRAIN_HIDDEN
HEADS = V1.PRETRAIN_HEADS
STATE_DIM = V1.PRETRAIN_STATE
WINDOW = 20
SEQ_LEN = 128
BATCH_SIZE = 32
EPOCHS = 20
WEIGHT_DECAY = 0.1
MAX_VAL_TOKENS = 500_000
SAVE_CKPT = True

LR_GRID = list(np.geomspace(3e-4, 5e-3, 6))
LR_SEARCH_EPOCHS = 2      # as in synthetic_lang_sl2.py

SUBSET = ["pure_tf", "pure_ssm", "4s4t_alternate", "6s2t_start"]
SEEDS = [0, 1, 2]
SOURCES = ["sl2", "none"]

RESULTS_ROOT = os.path.join(_REPO_ROOT, "results", "synthetic", "prepretrain")
DEFAULT_STAGE1_EPOCHS = V1.PRETRAIN_EPOCHS
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# stage 1: synthetic-language checkpoints
class Stage1:
    """Loads (or trains) the stage-1 model for one source; the corpus is generated once, lazily."""

    def __init__(self, source, ckpt_root=None, epochs=None):
        """epochs: stage-1 passes over the corpus; None = V1.PRETRAIN_EPOCHS with early stopping;
        values below 1 train on that fraction of one pass."""
        self.source = source
        self.epochs = epochs
        self.lang_dir = os.path.join(_LANGS_DIR, source)
        tag = "" if epochs is None else f"_ep{epochs:g}"
        if source == "sl2":
            self.ckpt_root = ckpt_root or os.path.join(V1.RESULTS_DIR, "pretrained" + tag)
            lr_file = os.path.join(_REPO_ROOT, "results", "synthetic", "lang_sl2", "auto_lr.json")
        else:
            self.ckpt_root = ckpt_root or os.path.join(V1.RESULTS_DIR, f"pretrained_{source}" + tag)
            lr_file = os.path.join(_REPO_ROOT, "results", "synthetic", "lang_sl2_sweep", source, "auto_lr.json")
        self.lrs = json.load(open(lr_file)) if os.path.exists(lr_file) else {}
        if not os.path.isdir(self.lang_dir):
            raise SystemExit(f"language directory not found: {self.lang_dir}")
        self.sl = SL2(load=True, dirname=self.lang_dir)
        self.tok = WordTokenizer(self.sl)
        self._train = self._val = None
        print(f"stage-1 source {source}: vocab={len(self.tok)} checkpoints={self.ckpt_root} "
              f"lrs={'from ' + lr_file if self.lrs else 'default 3e-4'}", flush=True)

    def lr(self, config_name):
        return self.lrs.get(config_name, 3e-4)

    def corpus(self):
        if self._train is None:
            print(f"  generating {self.source} corpus ({V1.N_TRAIN_SENTENCES}+{V1.N_VAL_SENTENCES} sentences)", flush=True)
            self._train = generate_corpus(self.sl, self.tok, V1.N_TRAIN_SENTENCES, V1.SENTENCE_LEN, seed=999)
            self._val = generate_corpus(self.sl, self.tok, V1.N_VAL_SENTENCES, V1.SENTENCE_LEN, seed=888)
        return self._train, self._val

    def model(self, config_name, seed):
        ckpt = os.path.join(self.ckpt_root, config_name, f"seed{seed}.pt")
        lr = self.lr(config_name)
        if os.path.exists(ckpt):
            m = get_model(V1.pretrain_args(config_name, lr), self.tok)
            m.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
            print(f"  stage-1 checkpoint loaded: {ckpt}", flush=True)
            return m.to(DEVICE).eval()
        if DEVICE != "cuda":
            raise SystemExit(f"stage-1 checkpoint missing and no GPU to train it: {ckpt}")
        set_all_seeds(seed)
        train, val = self.corpus()
        train_ds = None
        V1.PRETRAIN_EPOCHS = DEFAULT_STAGE1_EPOCHS
        if self.epochs is not None:
            if self.epochs < 1:
                V1.PRETRAIN_EPOCHS = 1
                train_ds = V1.LangDataset(train, V1.PRETRAIN_SEQ, V1.PRETRAIN_BATCH, shuffle=True)
                keep = max(1, int(round(self.epochs * len(train_ds)))) * V1.PRETRAIN_BATCH
                train_ds.input_ids = train_ds.input_ids[:keep]
                train_ds.output_ids = train_ds.output_ids[:keep]
                train_ds.mask = train_ds.mask[:keep]
            elif float(self.epochs).is_integer():
                V1.PRETRAIN_EPOCHS = int(self.epochs)
            else:
                raise SystemExit(f"--stage1-epochs must be < 1 or an integer, got {self.epochs}")
            print(f"  stage-1 budget: {self.epochs:g} epochs "
                  f"({len(train_ds) if train_ds is not None else 'full corpus'} batches/epoch)", flush=True)
        m, _ = V1.pretrain_model(config_name, lr, seed, train, val, self.tok, ckpt_root=self.ckpt_root,
                                 train_ds=train_ds)
        return m.eval()


# stage 2: BabyLM
def load_tokens_cached(tokenizer, split, max_tokens=None):
    """Cached GPT-2 token arrays from data/babylm/ or results/babylm/tokens/, if present."""
    name = f"{split}_tokens.npy"
    candidates = [os.path.join(_REPO_ROOT, "data", "babylm", name),
                  os.path.join(_REPO_ROOT, "results", "babylm", "tokens", name)]
    for cache_path in candidates:
        if os.path.exists(cache_path):
            toks = np.load(cache_path)
            if max_tokens is not None and len(toks) > max_tokens:
                toks = toks[:max_tokens]
            print(f"  [{split}] {len(toks):,} tokens (cached {os.path.relpath(cache_path, _REPO_ROOT)})", flush=True)
            return toks
    toks = load_babylm_tokens(tokenizer, split=split, max_tokens=max_tokens)
    if max_tokens is None:  # do not cache a truncated array
        os.makedirs(os.path.dirname(candidates[0]), exist_ok=True)
        np.save(candidates[0], toks)
        print(f"  [{split}] cached -> {candidates[0]}", flush=True)
    return toks


def babylm_args(config_name, lr, epochs):
    return Namespace(
        nope=False, layers=RE.layers_list(config_name),
        hidden_size=HIDDEN_SIZE, heads=HEADS, state_dim=STATE_DIM,
        lr=lr, epochs=epochs, sequence_length=SEQ_LEN, window=WINDOW,
        gradient_accumulation_steps=1, progress_bar=False,
        num_examples=999_999, num_log_steps=50,
    )


def window_mask():
    attn = torch.ones((SEQ_LEN, SEQ_LEN))
    return (torch.triu(attn, diagonal=0) - torch.triu(attn, diagonal=WINDOW)).T.to(DEVICE)


def eval_perplexity(model, dataset, attn):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for i in range(len(dataset)):
            batch = dataset[i]
            x = batch["input_ids"].to(DEVICE)
            y = batch["output_ids"].to(DEVICE)
            mask = batch["mask"].to(DEVICE)
            logits = model(x, attention_mask=attn, return_dict=True)["logits"]
            total_loss += ce_loss(y, logits, mask).item() * mask.sum().item()
            total_tokens += mask.sum().item()
    return math.exp(total_loss / total_tokens)


def build_model(config_name, lr, epochs, tokenizer, pre_model):
    model = get_model(babylm_args(config_name, lr, epochs), tokenizer)
    if pre_model is not None:
        transfer_backbone(pre_model, model)
    return model


def train_lm(args, model, train_ds, val_ds, max_epochs=None):
    """The LR schedule spans args.epochs; max_epochs stops early (used by the LR search)."""
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    accelerator = Accelerator(cpu=DEVICE == "cpu")
    model, optimizer = accelerator.prepare(model, optimizer)
    lr_scheduler = custom_get_scheduler(optimizer, len(train_ds) * args.epochs)
    attn = window_mask()
    val_ppls = []
    for epoch in range(max_epochs or args.epochs):
        model.train()
        avg_loss = train_epoch(model, epoch, accelerator, optimizer, lr_scheduler, attn, train_ds, args,
                               device=DEVICE)
        ppl = eval_perplexity(model, val_ds, attn)
        val_ppls.append(ppl)
        print(f"  epoch {epoch:2d}: train_loss={avg_loss:.4f}  val_ppl={ppl:.1f}", flush=True)
    return val_ppls, accelerator.unwrap_model(model)


def results_dir(source):
    return os.path.join(RESULTS_ROOT, source)


def find_best_lr(config_name, source, tokenizer, train_tokens, val_tokens, pre_model):
    cache_path = os.path.join(results_dir(source), "auto_lr.json")
    cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}
    if config_name in cache:
        return cache[config_name]
    print(f"  LR search for {config_name} ({source}): {len(LR_GRID)} points x {LR_SEARCH_EPOCHS} epoch", flush=True)
    ppls = {}
    for lr in LR_GRID:
        set_all_seeds(0)
        train_ds = TextDataset(train_tokens, SEQ_LEN, BATCH_SIZE, shuffle=True)
        val_ds = TextDataset(val_tokens, SEQ_LEN, BATCH_SIZE, shuffle=False)
        # full-length schedule, first LR_SEARCH_EPOCHS epochs only
        model = build_model(config_name, lr, EPOCHS, tokenizer, pre_model)
        val_ppls, _ = train_lm(babylm_args(config_name, lr, EPOCHS), model, train_ds, val_ds,
                               max_epochs=LR_SEARCH_EPOCHS)
        ppls[lr] = val_ppls[-1]
        print(f"    lr={lr:.2e}: ppl={val_ppls[-1]:.1f}", flush=True)
        del model; gc.collect(); torch.cuda.empty_cache()
    best = min(ppls, key=ppls.get)
    cache[config_name] = best
    os.makedirs(results_dir(source), exist_ok=True)
    json.dump(cache, open(cache_path, "w"), indent=2)
    print(f"  -> lr={best:.2e}", flush=True)
    return best


def run_single(config_name, seed, name, lr, tokenizer, train_tokens, val_tokens, pre_model, source, stage1_epochs=None):
    rdir = results_dir(name)
    rpath = os.path.join(rdir, config_name, f"{config_name}__seed{seed}.json")
    if os.path.exists(rpath):
        print(f"  SKIP {config_name} seed={seed} (exists)", flush=True)
        return
    set_all_seeds(seed)
    train_ds = TextDataset(train_tokens, SEQ_LEN, BATCH_SIZE, shuffle=True)
    val_ds = TextDataset(val_tokens, SEQ_LEN, BATCH_SIZE, shuffle=False)
    args = babylm_args(config_name, lr, EPOCHS)
    model = build_model(config_name, lr, EPOCHS, tokenizer, pre_model)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  {config_name} seed={seed} lr={lr:.2e}: {n_params:,} params, "
          f"{len(train_ds)} train batches, {len(val_ds)} val batches", flush=True)

    val_ppls, model = train_lm(args, model, train_ds, val_ds)

    ckpt = os.path.join(rdir, "checkpoints", config_name, f"seed{seed}.pt")
    if SAVE_CKPT:
        os.makedirs(os.path.dirname(ckpt), exist_ok=True)
        torch.save({k: v.cpu() for k, v in model.state_dict().items()}, ckpt)

    os.makedirs(os.path.dirname(rpath), exist_ok=True)
    with open(rpath, "w") as f:
        json.dump({
            "config": config_name, "layers": RE.CONFIGS[config_name], "seed": seed,
            "source": source, "pretrained": pre_model is not None,
            "stage1_epochs": 0 if pre_model is None else stage1_epochs,
            "hidden_size": HIDDEN_SIZE, "window": WINDOW, "seq_len": SEQ_LEN,
            "batch_size": BATCH_SIZE, "lr": lr, "epochs": EPOCHS, "weight_decay": WEIGHT_DECAY,
            "n_params": n_params, "val_ppls": val_ppls,
            "final_ppl": val_ppls[-1], "best_ppl": min(val_ppls),
            "checkpoint": None if not SAVE_CKPT else
                          os.path.relpath(ckpt, _REPO_ROOT) if ckpt.startswith(_REPO_ROOT) else ckpt,
        }, f, indent=2)
    print(f"  -> saved {rpath}", flush=True)
    del model; gc.collect(); torch.cuda.empty_cache()


# main
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="*", default=SOURCES, help="langs/<name> or none (scratch control)")
    ap.add_argument("--configs", nargs="*", default=SUBSET)
    ap.add_argument("--seeds", nargs="*", type=int, default=SEEDS)
    ap.add_argument("--stage1-root", default=None, help="override the stage-1 checkpoint directory")
    ap.add_argument("--stage1-epochs", type=float, default=None,
                    help="stage-1 budget in SL2 epochs (<1 = that fraction of one epoch); default 15 with early stopping")
    ap.add_argument("--lr", nargs="+", default=None,
                    help="fixed LR, skips the search: one value for all configs or config=value pairs")
    ap.add_argument("--results-name", default=None,
                    help="results/checkpoints subdirectory under results/synthetic/prepretrain (default: the source name)")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--max-train-tokens", type=int, default=None, help="smoke tests")
    ap.add_argument("--max-val-tokens", type=int, default=MAX_VAL_TOKENS)
    ap.add_argument("--results-dir", default=None)
    ap.add_argument("--window", type=int, default=None,
                    help="attention window of the T layers (default %d; >= %d is full causal attention); a different "
                         "value writes to <source>_w<window>" % (WINDOW, SEQ_LEN))
    ap.add_argument("--no-ckpt", action="store_true", help="do not save the final weights")
    ap.add_argument("--search-only", action="store_true",
                    help="run the LR search (written to <results>/auto_lr.json) and skip the training runs")
    a = ap.parse_args()
    if a.window: WINDOW = a.window
    if a.no_ckpt: SAVE_CKPT = False
    if a.epochs: EPOCHS = a.epochs
    if a.results_dir: RESULTS_ROOT = os.path.abspath(a.results_dir)
    fixed_lr = {}
    for tok in a.lr or []:
        cfg, _, val = tok.rpartition("=")
        fixed_lr[cfg or "*"] = float(val)

    print("=== Pre-pre-training stage 2: synthetic language -> BabyLM ===", flush=True)
    print(f"sources={a.sources} configs={a.configs} seeds={a.seeds} device={DEVICE}", flush=True)
    print(f"d={HIDDEN_SIZE} window={WINDOW} seq={SEQ_LEN} batch={BATCH_SIZE} epochs={EPOCHS} wd={WEIGHT_DECAY} "
          f"LR grid {LR_GRID[0]:.0e}..{LR_GRID[-1]:.0e}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    print("\nLoading BabyLM...", flush=True)
    train_tokens = load_tokens_cached(tokenizer, "train", a.max_train_tokens)
    val_tokens = load_tokens_cached(tokenizer, "validation", a.max_val_tokens)

    for source in a.sources:
        print(f"\n{'=' * 60}\n=== source: {source} ===\n{'=' * 60}", flush=True)
        stage1 = None if source == "none" else Stage1(source, a.stage1_root, a.stage1_epochs)
        name = a.results_name or (source if WINDOW == 20 else f"{source}_w{WINDOW}")
        for cfg in a.configs:
            pre0 = stage1.model(cfg, a.seeds[0]) if stage1 else None
            lr = fixed_lr.get(cfg, fixed_lr.get("*")) or find_best_lr(cfg, name, tokenizer, train_tokens, val_tokens, pre0)
            if a.search_only:
                del pre0
                continue
            for seed in a.seeds:
                print(f"\n=== {name} / {cfg} seed={seed} ===", flush=True)
                pre = pre0 if (stage1 and seed == a.seeds[0]) else (stage1.model(cfg, seed) if stage1 else None)
                run_single(cfg, seed, name, lr, tokenizer, train_tokens, val_tokens, pre, source,
                           a.stage1_epochs if a.stage1_epochs is not None else DEFAULT_STAGE1_EPOCHS)
                del pre; gc.collect(); torch.cuda.empty_cache()
            del pre0
    print("\n=== stage 2 done ===", flush=True)
