#!/usr/bin/env python3
"""Pre-train the hybrids on SL2, then fine-tune on MKAR and MQAR.

Pre-training: d=256, window 100, sequence length 200, 15 epochs, early stopping (patience 5).
Post-training: d=256, window 20, 48 epochs. The pre-trained checkpoints are shared with
posttrain_sl2_v2.py and pretrain_babylm.py.

Results -> results/synthetic/posttrain_sl2/
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import os, sys, gc, json, math, random, time, types
import numpy as np
import torch
import torch.nn.functional as F
from argparse import Namespace
from torch.optim import AdamW
from accelerate import Accelerator

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import decode_recall as RE
from model_utils import get_model
from data_utils import get_tokenizer, get_train_dataset
from train_utils import ce_loss, custom_get_scheduler, train, train_epoch
from test_utils import evaluation

_LANGS_DIR = os.path.join(_HERE, "..", "..", "langs")
sys.path.insert(0, _LANGS_DIR)
from sl2 import SL2, SL2_Config

# hyperparameters
PRETRAIN_HIDDEN = 256
PRETRAIN_WINDOW = 100
PRETRAIN_SEQ = 200
PRETRAIN_BATCH = 32
PRETRAIN_EPOCHS = 15
EARLY_STOP_PATIENCE = 5
PRETRAIN_HEADS = 1
PRETRAIN_STATE = 1
N_TRAIN_SENTENCES = 20000
N_VAL_SENTENCES = 2000
SENTENCE_LEN = 256

POSTTRAIN_WINDOW = 20
POSTTRAIN_EPOCHS = 48

RESULTS_DIR = os.path.abspath(os.path.join(
    _HERE, "..", "..", "results", "synthetic", "posttrain_sl2"))

SUBSET = ["pure_tf", "pure_ssm", "4s4t_alternate", "6s2t_start"]
SEEDS = [0, 1, 2, 3, 4]
TASKS = ["mkar", "mqar"]

LR_GRID = list(np.geomspace(1e-5, 3e-3, 7))
LR_ITERS = 2


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# SL2 language
# saved language in langs/sl2; raises if missing
PRETRAIN_LANG = "sl2"


def load_language():
    _prev = os.getcwd()
    os.chdir(_LANGS_DIR)
    try:
        sl = SL2(load=True, dirname=PRETRAIN_LANG)
    finally:
        os.chdir(_prev)
    return sl


class WordTokenizer:
    def __init__(self, sl):
        self.word2id = dict(sl.vtoi)
        self.id2word = list(sl.config.all_vocab)
        self.vocab_size = len(self.id2word)

    def encode(self, text):
        return [self.word2id[w] for w in text.split() if w in self.word2id]

    def __len__(self):
        return self.vocab_size


class LangDataset:
    def __init__(self, sequences, seq_len, batch_size, shuffle=True):
        flat = []
        for s in sequences:
            flat.extend(s)
        flat = torch.tensor(flat, dtype=torch.long)
        n_chunks = len(flat) // (seq_len + 1)
        flat = flat[:n_chunks * (seq_len + 1)]
        chunks = flat.view(n_chunks, seq_len + 1)
        self.input_ids = chunks[:, :-1]
        self.output_ids = chunks[:, 1:]
        self.mask = torch.ones_like(self.output_ids, dtype=torch.float32)
        self.batch_size = batch_size
        if shuffle:
            perm = torch.randperm(len(self.input_ids))
            self.input_ids = self.input_ids[perm]
            self.output_ids = self.output_ids[perm]

    def __len__(self):
        return len(self.input_ids) // self.batch_size

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError(idx)
        s = idx * self.batch_size
        e = s + self.batch_size
        return {
            "input_ids": self.input_ids[s:e],
            "output_ids": self.output_ids[s:e],
            "mask": self.mask[s:e],
        }


def generate_corpus(sl, tok, n_sentences, sentence_len, seed=42):
    rng_state = random.getstate()
    np_state = np.random.get_state()
    random.seed(seed)
    np.random.seed(seed)
    sequences = []
    for _ in range(n_sentences):
        text = sl.generate_sentence(sentence_len)
        ids = tok.encode(text)
        sequences.append(ids)
    random.setstate(rng_state)
    np.random.set_state(np_state)
    return sequences


# pre-training
def pretrain_args(config_name, lr):
    layers = RE.layers_list(config_name)
    return Namespace(
        nope=False, layers=layers,
        hidden_size=PRETRAIN_HIDDEN, heads=PRETRAIN_HEADS,
        state_dim=PRETRAIN_STATE, lr=lr, epochs=PRETRAIN_EPOCHS,
        sequence_length=PRETRAIN_SEQ, window=PRETRAIN_WINDOW,
        gradient_accumulation_steps=1, progress_bar=False,
        num_examples=999_999, num_log_steps=50,
    )


def eval_perplexity(model, dataset, attn_mask):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for i in range(len(dataset)):
            batch = dataset[i]
            x = batch["input_ids"].to("cuda")
            y = batch["output_ids"].to("cuda")
            mask = batch["mask"].to("cuda")
            logits = model(x, attention_mask=attn_mask, return_dict=True)["logits"]
            loss = ce_loss(y, logits, mask)
            n = mask.sum().item()
            total_loss += loss.item() * n
            total_tokens += n
    return math.exp(total_loss / total_tokens)


def pretrain_model(config_name, lr, seed, train_sequences, val_sequences, tok, ckpt_root=None, train_ds=None):
    """ckpt_root: directory holding <config>/seed<k>.pt (default results/synthetic/posttrain_sl2/pretrained).
    train_ds: prebuilt batch dataset (LangDataset-like) used instead of chunking train_sequences."""
    ckpt_dir = os.path.join(ckpt_root or os.path.join(RESULTS_DIR, "pretrained"), config_name)
    ckpt_path = os.path.join(ckpt_dir, f"seed{seed}.pt")

    args = pretrain_args(config_name, lr)
    model = get_model(args, tok)
    n_params = sum(p.numel() for p in model.parameters())

    if os.path.exists(ckpt_path):
        print(f"  Loading pre-trained checkpoint: {ckpt_path}", flush=True)
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True))
        model = model.to("cuda")
        return model, n_params

    print(f"  Pre-training {config_name} seed={seed} lr={lr:.2e}: {n_params:,} params", flush=True)

    if train_ds is None:
        train_ds = LangDataset(train_sequences, PRETRAIN_SEQ, PRETRAIN_BATCH, shuffle=True)
    val_ds = LangDataset(val_sequences, PRETRAIN_SEQ, PRETRAIN_BATCH, shuffle=False)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    accelerator = Accelerator()
    model, optimizer = accelerator.prepare(model, optimizer)

    total_steps = len(train_ds) * args.epochs
    lr_scheduler = custom_get_scheduler(optimizer, total_steps)

    attn = torch.ones((PRETRAIN_SEQ, PRETRAIN_SEQ))
    attn = (torch.triu(attn, diagonal=0) -
            torch.triu(attn, diagonal=PRETRAIN_WINDOW)).T.to("cuda")

    best_ppl = float("inf")
    wait = 0
    for epoch in range(args.epochs):
        model.train()
        avg_loss = train_epoch(model, epoch, accelerator, optimizer,
                               lr_scheduler, attn, train_ds, args)
        ppl = eval_perplexity(model, val_ds, attn)

        if ppl < best_ppl - 0.01:
            best_ppl = ppl
            wait = 0
        else:
            wait += 1

        if (epoch + 1) % 3 == 0 or epoch == args.epochs - 1 or wait >= EARLY_STOP_PATIENCE:
            print(f"    pretrain ep {epoch:2d}: loss={avg_loss:.4f} ppl={ppl:.1f}", flush=True)

        if wait >= EARLY_STOP_PATIENCE:
            print(f"    early stop at epoch {epoch} (no improvement for {wait} epochs)", flush=True)
            break

    unwrapped = accelerator.unwrap_model(model)
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save(unwrapped.state_dict(), ckpt_path)
    print(f"    -> saved {ckpt_path} (best_ppl={best_ppl:.2f})", flush=True)

    del optimizer, accelerator, lr_scheduler, train_ds, val_ds
    gc.collect()
    torch.cuda.empty_cache()

    return unwrapped, n_params


# post-training on MKAR / MQAR
def transfer_backbone(pretrained_model, target_model):
    pretrained_sd = pretrained_model.state_dict()
    target_sd = target_model.state_dict()
    transferred, skipped = 0, 0
    for name, param in pretrained_sd.items():
        if name in target_sd and target_sd[name].shape == param.shape:
            target_sd[name].copy_(param)
            transferred += 1
        else:
            skipped += 1
    target_model.load_state_dict(target_sd)
    print(f"    Transferred {transferred} params, skipped {skipped}", flush=True)
    return target_model


def task_args(config_name, lr, task):
    layers = RE.layers_list(config_name)
    from generate import force_args
    a = Namespace(
        train_task=task, eval_task=task,
        num_vocab=RE.DEFAULTS["num_vocab"], num_numbers=2, min_number=0,
        p=0.2, eval_p=0.2, mixed=False, ood_eval=False,
        nope=False, model=None, num_layers=None,
        layer1=None, layer2=None, layer3=None,
        layers=layers,
        hidden_size=PRETRAIN_HIDDEN, heads=PRETRAIN_HEADS,
        num_masked_heads=1, state_dim=PRETRAIN_STATE,
        lr=lr, auto_lr=False, force_do_lr=False,
        epochs=POSTTRAIN_EPOCHS, num_examples=RE.DEFAULTS["num_examples"],
        num_eval_examples=100, window=POSTTRAIN_WINDOW,
        train_batch_size=RE.DEFAULTS["train_batch_size"],
        eval_batch_size=RE.DEFAULTS["eval_batch_size"], eval_num_batches=1,
        pack_examples=False,
        min_train_length=RE.DEFAULTS.get("min_train_length", 97),
        max_train_length=RE.DEFAULTS.get("max_train_length", 98),
        min_eval_length=RE.DEFAULTS.get("min_eval_length", 97),
        max_eval_length=RE.DEFAULTS.get("max_eval_length", 98),
        gradient_accumulation_steps=1,
        sequence_length=RE.DEFAULTS["sequence_length"],
        eval_sequence_length=RE.DEFAULTS["sequence_length"],
        save_model=False, save_results=False, run_anyways=False,
        run_number=-1,
        print=False, progress_bar=False, num_log_steps=50, test_generate=False,
    )
    force_args(a)
    return a


def find_best_lr_posttrain(config_name, pretrained_model, task):
    cache_path = os.path.join(RESULTS_DIR, f"auto_lr_{task}.json")
    if os.path.exists(cache_path):
        try:
            cache = json.load(open(cache_path))
            if config_name in cache:
                return cache[config_name]
        except (json.JSONDecodeError, OSError):
            pass
    else:
        cache = {}

    losses = []
    for it in range(LR_ITERS):
        for lr in LR_GRID:
            set_all_seeds(it)
            args = task_args(config_name, lr, task)
            tokenizer = get_tokenizer(args)
            train_dataset = get_train_dataset(args, tokenizer)
            model = get_model(args, tokenizer)
            model = transfer_backbone(pretrained_model, model)
            accs, final_loss = train(args, model, tokenizer, train_dataset, one_epoch=True)
            losses.append((float(final_loss), float(lr)))
            del model, tokenizer, train_dataset
            gc.collect(); torch.cuda.empty_cache()

    losses.sort(key=lambda t: t[0])
    best = losses[0][1]
    cache[config_name] = best
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(cache, f, indent=2)
    print(f"  auto-LR (post-train {task}) {config_name}: best={best:.2e}", flush=True)
    return best


def posttrain_single(config_name, pretrain_lr, posttrain_lr, seed,
                     train_sequences, val_sequences, tok, task):
    task_dir = os.path.join(RESULTS_DIR, task)
    rpath = os.path.join(task_dir, config_name,
                         f"{config_name}__seed{seed}.json")
    if os.path.exists(rpath):
        print(f"  SKIP {task} {config_name} seed={seed} (exists)", flush=True)
        return

    set_all_seeds(seed)

    print(f"\n[{task} {config_name} seed={seed}] Phase 1: Pre-training on SL2", flush=True)
    t0 = time.time()
    pretrained_model, pretrain_params = pretrain_model(
        config_name, pretrain_lr, seed, train_sequences, val_sequences, tok)
    pretrain_time = time.time() - t0

    print(f"[{task} {config_name} seed={seed}] Phase 2: Post-training on {task.upper()} "
          f"(lr={posttrain_lr:.2e})", flush=True)
    set_all_seeds(seed)

    args = task_args(config_name, posttrain_lr, task)
    tokenizer = get_tokenizer(args)
    train_dataset = get_train_dataset(args, tokenizer)

    model = get_model(args, tokenizer)
    posttrain_params = sum(p.numel() for p in model.parameters())
    model = transfer_backbone(pretrained_model, model)

    del pretrained_model
    gc.collect(); torch.cuda.empty_cache()

    t1 = time.time()
    accs, final_loss, train_losses = train(args, model, tokenizer, train_dataset,
                                            return_loss_history=True)
    model.eval()
    str_mean, str_std, char_acc = evaluation(args, model, tokenizer, do_print=False)
    posttrain_time = time.time() - t1

    result = {
        "config": config_name,
        "architecture": RE.CONFIGS[config_name],
        "seed": seed,
        "task": task,
        "pretrain_source": "sl2",
        "pretrain_lr": pretrain_lr,
        "posttrain_lr": posttrain_lr,
        "pretrain_epochs": PRETRAIN_EPOCHS,
        "posttrain_epochs": POSTTRAIN_EPOCHS,
        "pretrain_params": pretrain_params,
        "posttrain_params": posttrain_params,
        "pretrain_hidden": PRETRAIN_HIDDEN,
        "pretrain_window": PRETRAIN_WINDOW,
        "pretrain_seq": PRETRAIN_SEQ,
        "posttrain_window": POSTTRAIN_WINDOW,
        "pretrain_time_s": pretrain_time,
        "posttrain_time_s": posttrain_time,
        "train_accs": accs,
        "train_losses": train_losses,
        "final_acc": char_acc,
        "mean_acc": float(np.mean(char_acc)),
        "str_acc": str_mean,
        "final_loss": final_loss,
    }

    os.makedirs(os.path.dirname(rpath), exist_ok=True)
    with open(rpath, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  -> mean_acc={result['mean_acc']:.4f} str_acc={np.mean(str_mean):.4f} "
          f"({pretrain_time:.0f}s pretrain + {posttrain_time:.0f}s posttrain)", flush=True)

    del model, tokenizer, train_dataset
    gc.collect(); torch.cuda.empty_cache()


# main
if __name__ == "__main__":
    print(f"=== Post-training: SL2 -> MKAR/MQAR ===", flush=True)
    print(f"Pre-train: SL2 d={PRETRAIN_HIDDEN} w={PRETRAIN_WINDOW} seq={PRETRAIN_SEQ} "
          f"E={PRETRAIN_EPOCHS} (early stop patience={EARLY_STOP_PATIENCE})", flush=True)
    print(f"Post-train: MKAR/MQAR w={POSTTRAIN_WINDOW} E={POSTTRAIN_EPOCHS}", flush=True)
    print(f"Configs: {SUBSET}  Seeds: {SEEDS}  Tasks: {TASKS}", flush=True)

    sl = load_language()
    tok = WordTokenizer(sl)
    print(f"\nSL2 vocab={len(tok)} (N_GRAMMAR={sl.config.N_GRAMMAR}, "
          f"MULTI_FACT={sl.config.MULTI_FACT}, USE_RECALL={sl.config.USE_RECALL})", flush=True)

    print(f"\nGenerating SL2 corpus...", flush=True)
    train_sequences = generate_corpus(sl, tok, N_TRAIN_SENTENCES, SENTENCE_LEN, seed=999)
    val_sequences = generate_corpus(sl, tok, N_VAL_SENTENCES, SENTENCE_LEN, seed=888)
    print(f"  train={sum(len(s) for s in train_sequences):,} tokens, "
          f"val={sum(len(s) for s in val_sequences):,} tokens", flush=True)

    sl2_lr_cache = os.path.join(
        _HERE, "..", "..", "results", "synthetic", "lang_sl2", "auto_lr.json")
    if os.path.exists(sl2_lr_cache):
        pretrain_lrs = json.load(open(sl2_lr_cache))
        print(f"\nLoaded pre-train LRs from SL2 experiment: {sl2_lr_cache}", flush=True)
    else:
        pretrain_lrs = {c: 3e-4 for c in SUBSET}
        print(f"\nUsing default pre-train LR=3e-4", flush=True)
    for c in SUBSET:
        print(f"  {c:20s} pretrain_lr={pretrain_lrs.get(c, 3e-4):.2e}", flush=True)

    # pre-train seed-0 models for the LR search
    print(f"\nStep 1: Pre-train seed=0 models for auto-LR search", flush=True)
    pretrained_seed0 = {}
    for config in SUBSET:
        set_all_seeds(0)
        lr = pretrain_lrs.get(config, 3e-4)
        model, _ = pretrain_model(config, lr, 0, train_sequences, val_sequences, tok)
        pretrained_seed0[config] = model

    # LR search per task
    posttrain_lrs = {}
    for task in TASKS:
        print(f"\nStep 2: Auto-LR search for {task.upper()} post-training", flush=True)
        posttrain_lrs[task] = {}
        for config in SUBSET:
            posttrain_lrs[task][config] = find_best_lr_posttrain(
                config, pretrained_seed0[config], task)
        print(f"\nResolved {task.upper()} post-train LRs:", flush=True)
        for c in SUBSET:
            print(f"  {c:20s} posttrain_lr={posttrain_lrs[task][c]:.2e}", flush=True)

    del pretrained_seed0
    gc.collect(); torch.cuda.empty_cache()

    total_runs = len(SUBSET) * len(SEEDS) * len(TASKS)
    print(f"\nStep 3: {total_runs} post-training runs", flush=True)
    for task in TASKS:
        print(f"\n{'='*60}", flush=True)
        print(f"=== {task.upper()} post-training ===", flush=True)
        print(f"{'='*60}", flush=True)
        for config in SUBSET:
            for seed in SEEDS:
                posttrain_single(
                    config,
                    pretrain_lr=pretrain_lrs.get(config, 3e-4),
                    posttrain_lr=posttrain_lrs[task][config],
                    seed=seed,
                    train_sequences=train_sequences,
                    val_sequences=val_sequences,
                    tok=tok,
                    task=task,
                )

    print("\n=== Post-training (SL2 -> MKAR/MQAR) complete ===", flush=True)
