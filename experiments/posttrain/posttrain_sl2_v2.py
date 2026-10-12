#!/usr/bin/env python3
"""Post-training on MKAR and MQAR from an SL2-pre-trained backbone, with monitors of how much the
backbone changes.

Variants
  full    all backbone params trainable, wd=0.01, LR grid 1e-5..3e-4
  lora    backbone frozen, LoRA(r=8, alpha=16) on every nn.Linear in the layers,
          fresh embed_in / embed_out + final LN trainable, wd=0.01, LR grid 1e-4..3e-3
  scratch no pre-training, same recipe as the d=256 sweeps (wd=0.1, LR grid 1e-4..1e-3)
  scratch_matched  no pre-training, post-training recipe (wd=0.01, LR grid 1e-5..3e-4)
  shared  full, with the task written in the SL2 vocabulary
  mixed   shared, pre-trained on SL2 plus task sequences

Monitors: per-epoch backbone drift per layer, KL(P_pre || P_post) on SL2, and the SL2 perplexity
of the post-trained backbone.

Pre-trained checkpoints are shared with posttrain_sl2.py:
  results/synthetic/posttrain_sl2/pretrained/<config>/seed<k>.pt
Results -> results/synthetic/posttrain_sl2_v2/<task>/<variant>/<config>/<config>__seed<k>.json
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import os, sys, gc, json, math, time, argparse, random
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from accelerate import Accelerator

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import decode_recall as RE
from model_utils import get_model
from data_utils import get_tokenizer, get_train_dataset, Tokenizer
from generate import generate_seq
from train_utils import ce_loss, custom_get_scheduler, train_epoch
from test_utils import evaluation

import posttrain_sl2 as V1
from posttrain_sl2 import (set_all_seeds, load_language, LangDataset, generate_corpus,
                           pretrain_model, pretrain_args, transfer_backbone, task_args)
from lora import apply_lora, effective_state_dict, trainable_parameters, count_params

_SL_DIR = os.path.join(_HERE, "..", "synth_lang")
sys.path.insert(0, _SL_DIR)
from synthetic_lang_sl2 import WordTokenizer as SL2Tokenizer  # has token_type_masks

# hyperparameters
RESULTS_DIR = os.path.abspath(os.path.join(
    _HERE, "..", "..", "results", "synthetic", "posttrain_sl2_v2"))
CKPT_DIR = os.path.join(RESULTS_DIR, "checkpoints")

SUBSET = ["pure_tf", "pure_ssm", "4s4t_alternate", "6s2t_start"]
SEEDS = [0, 1, 2, 3, 4]
TASKS = ["mkar", "mqar"]
VARIANTS = ["full", "lora"]

POSTTRAIN_WD = 0.01          # lower than pre-training (0.1)
SCRATCH_WD = 0.1             # matches train_utils.train
# variants without pre-training; scratch_matched uses the post-training recipe (LR grid, POSTTRAIN_WD)
NO_PRETRAIN = ("scratch", "scratch_matched")
LORA_R = 8
LORA_ALPHA = 16.0

LR_GRIDS = {
    "full":    list(np.geomspace(1e-5, 3e-4, 7)),
    "shared":  list(np.geomspace(1e-5, 3e-4, 7)),   # task embedded in the SL2 vocabulary
    "mixed":   list(np.geomspace(1e-5, 3e-4, 7)),   # as shared, pre-trained on SL2 + task sequences
    "lora":    list(np.geomspace(1e-4, 3e-3, 7)),   # adapters tolerate higher LR
    "scratch": list(RE.LR_GRID),                     # 1e-4..1e-3, as in the d256 sweeps
    "scratch_matched": list(np.geomspace(1e-5, 3e-4, 7)),
}
LR_ITERS = 2
LR_SEARCH_EPOCHS = 2

N_PROBE_SENTENCES = 320      # SL2 val sentences used for the KL / PPL monitor
PROBE_EVERY = 1              # epochs between monitor evaluations


# Variant "shared": the task is written in the SL2 vocabulary (values V0..V31 -> the first 32
# non-grammar words, <bos> -> <start>, <eos> / <null> / separators -> the first grammar words), so
# embed_in / embed_out transfer with the backbone.

TOK_SL2 = None      # set in main; the SL2 WordTokenizer the shared tokenizer maps into


class SharedVocabTokenizer(Tokenizer):
    """Map task tokens into the SL2 vocabulary: <bos> -> <start>, <eos> / <null> / #0 -> grammar
    words, V0.. -> non-grammar words."""
    def __init__(self, base, tok_sl2):
        ng = sorted(tok_sl2.non_grammar_ids); g = sorted(tok_sl2.grammar_ids)
        nums = list(base.number_tokens)
        specials = ["<eos>", "<null>"] + nums[:1]
        extra = nums[1:]
        nv = len(base.vocab_tokens)
        assert nv + len(extra) <= len(ng), "task values + extra number tokens exceed the SL2 non-grammar words"
        assert len(specials) <= len(g)
        to_token = {v: ng[i] for i, v in enumerate(base.vocab_tokens)}
        to_token.update({t: ng[nv + i] for i, t in enumerate(extra)})
        to_token["<bos>"] = tok_sl2.word2id["<start>"]
        for i, sp in enumerate(specials):
            to_token[sp] = g[i]
        super().__init__(to_token, base.vocab_tokens, base.number_tokens)
        self.sl2_vocab_size = len(tok_sl2)
        self.token_map = {k: tok_sl2.id2word[v] for k, v in to_token.items()}

    def __len__(self):
        return self.sl2_vocab_size


# Variant "mixed": pre-train on SL2 sentences plus task sequences in the SL2 vocabulary, then
# post-train as "shared". Checkpoints: results/synthetic/posttrain_sl2/pretrained_mixed/.

MIX_FRAC = 0.20                 # fraction of pre-training tokens from task sequences
MIX_TASKS = ["mkar", "mqar"]
MIX_SEQ_LEN = 98                # = RE.DEFAULTS max_train_length
PRETRAINED_MIXED_DIR = os.path.join(V1.RESULTS_DIR, "pretrained_mixed")


def task_pairs(task, n, tok_sl2, seed):
    """n sequences of `task` in the post-training format, as SL2 token ids: (input_ids, target_ids,
    mask). Targets are the value at <null> placeholders and the next token elsewhere; mask is 0
    on trailing padding."""
    args = task_args("pure_tf", 1e-4, task)
    tokenizer = SharedVocabTokenizer(get_tokenizer(args), tok_sl2)
    null = "<null>"
    st = np.random.get_state(); np.random.seed(seed)
    out = []
    for _ in range(n):
        inp, tgt = generate_seq(tokenizer, MIX_SEQ_LEN, task, p=args.p, mixed=False)
        L = len(inp)
        while L > 0 and inp[L - 1] == null and tgt[L - 1] == null:
            L -= 1                                  # trailing padding
        x = tokenizer(inp[:L]).tolist()
        y, m = [], []
        for i in range(L):
            if tgt[i] != null:
                y.append(tokenizer.TO_TOKEN[tgt[i]]); m.append(1.0)
            elif i + 1 < L:
                y.append(x[i + 1]); m.append(1.0)
            else:
                y.append(x[i]); m.append(0.0)
        out.append((x, y, m))
    np.random.set_state(st)
    return out


class MixedDataset:
    """Batches of PRETRAIN_SEQ-token rows mixing SL2 chunks and packed task sequences."""

    def __init__(self, sl2_sequences, pairs, seq_len, batch_size, seed):
        base = LangDataset(sl2_sequences, seq_len, batch_size, shuffle=False)
        rows_x = [base.input_ids]; rows_y = [base.output_ids]; rows_m = [base.mask]
        xs, ys, ms = [], [], []
        cur_x, cur_y, cur_m = [], [], []
        for x, y, m in pairs:
            if len(cur_x) + len(x) > seq_len:
                xs.append(cur_x); ys.append(cur_y); ms.append(cur_m); cur_x, cur_y, cur_m = [], [], []
            cur_x += x; cur_y += y; cur_m += m
        if cur_x:
            xs.append(cur_x); ys.append(cur_y); ms.append(cur_m)
        pad = 0
        for x, y, m in zip(xs, ys, ms):
            k = seq_len - len(x)
            rows_x.append(torch.tensor([x + [pad] * k])); rows_y.append(torch.tensor([y + [pad] * k]))
            rows_m.append(torch.tensor([m + [0.0] * k], dtype=torch.float32))
        self.input_ids = torch.cat(rows_x); self.output_ids = torch.cat(rows_y); self.mask = torch.cat(rows_m)
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(len(self.input_ids), generator=g)
        self.input_ids, self.output_ids, self.mask = self.input_ids[perm], self.output_ids[perm], self.mask[perm]
        self.batch_size = batch_size
        self.n_task_rows = len(xs); self.n_sl2_rows = len(base.input_ids)

    def __len__(self):
        return len(self.input_ids) // self.batch_size

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError(idx)
        s = idx * self.batch_size; e = s + self.batch_size
        return {"input_ids": self.input_ids[s:e], "output_ids": self.output_ids[s:e], "mask": self.mask[s:e]}


def mixed_dataset(train_sequences, tok_sl2, seed):
    """SL2 sentences + task sequences (MIX_FRAC of the tokens, split evenly over MIX_TASKS)."""
    n_sl2_tokens = sum(len(s) for s in train_sequences)
    n_task_tokens = int(n_sl2_tokens * MIX_FRAC / (1 - MIX_FRAC))
    per_task = n_task_tokens // (len(MIX_TASKS) * MIX_SEQ_LEN)
    pairs = []
    for i, task in enumerate(MIX_TASKS):
        pairs += task_pairs(task, per_task, tok_sl2, seed * 100 + i)
    rng = random.Random(seed); rng.shuffle(pairs)
    ds = MixedDataset(train_sequences, pairs, V1.PRETRAIN_SEQ, V1.PRETRAIN_BATCH, seed)
    return ds, per_task


def pretrain_model_mixed(config_name, lr, seed, train_sequences, val_sequences, tok_sl2):
    ck = os.path.join(PRETRAINED_MIXED_DIR, config_name, f"seed{seed}.pt")
    if os.path.exists(ck):
        return V1.pretrain_model(config_name, lr, seed, None, val_sequences, tok_sl2, ckpt_root=PRETRAINED_MIXED_DIR)
    ds, per_task = mixed_dataset(train_sequences, tok_sl2, seed)
    print(f"  mixed corpus: {ds.n_sl2_rows} SL2 rows + {ds.n_task_rows} task rows "
          f"({per_task} x {len(MIX_TASKS)} task sequences, {V1.PRETRAIN_SEQ}-token rows)", flush=True)
    return V1.pretrain_model(config_name, lr, seed, train_sequences, val_sequences, tok_sl2,
                             ckpt_root=PRETRAINED_MIXED_DIR, train_ds=ds)


def get_pretrained(config_name, lr, seed, variant, train_sequences, val_sequences, tok_sl2):
    if variant == "mixed":
        return pretrain_model_mixed(config_name, lr, seed, train_sequences, val_sequences, tok_sl2)
    return V1.pretrain_model(config_name, lr, seed, train_sequences, val_sequences, tok_sl2)


# SL2 probe: KL(P_pre || P_post) and PPL on SL2
class SL2Probe:
    """Frozen pre-trained model plus a clone that takes the post-trained backbone (LoRA merged)
    with the pre-trained SL2 embeddings and head; KL(P_pre || P_post) measures backbone drift."""

    def __init__(self, config_name, pretrain_lr, pre_model, val_sequences, tok_sl2):
        self.tok = tok_sl2
        self.pre = pre_model.eval()
        self.pre_sd = {k: v.detach().clone() for k, v in pre_model.state_dict().items()}
        self.probe = get_model(pretrain_args(config_name, pretrain_lr), tok_sl2).to("cuda").eval()
        self.ds = LangDataset(val_sequences[:N_PROBE_SENTENCES], V1.PRETRAIN_SEQ,
                              V1.PRETRAIN_BATCH, shuffle=False)
        attn = torch.ones((V1.PRETRAIN_SEQ, V1.PRETRAIN_SEQ))
        self.attn = (torch.triu(attn, diagonal=0) -
                     torch.triu(attn, diagonal=V1.PRETRAIN_WINDOW)).T.to("cuda")

    @torch.no_grad()
    def evaluate(self, post_eff_sd, include_embeddings=False):
        """include_embeddings=False: post-trained backbone with the pre-trained SL2 embeddings/head.
        True: the whole post-trained model (shared-vocabulary variant only)."""
        sd = dict(self.pre_sd)
        for k, v in post_eff_sd.items():
            backbone = k.startswith("hybrid.layers.") or k.startswith("hybrid.final_layer_norm.")
            if backbone or include_embeddings:
                if k in sd and sd[k].shape == v.shape:
                    sd[k] = v
        self.probe.load_state_dict(sd)

        tot_kl, tot_nll, n = 0.0, 0.0, 0
        kl_t = {"grammar": 0.0, "non_grammar": 0.0, "fact_recall": 0.0}
        n_t = {"grammar": 0, "non_grammar": 0, "fact_recall": 0}
        for i in range(len(self.ds)):
            b = self.ds[i]
            x = b["input_ids"].to("cuda"); y = b["output_ids"].to("cuda")
            lp = F.log_softmax(self.pre(x, attention_mask=self.attn, return_dict=True)["logits"].float(), -1)
            lq = F.log_softmax(self.probe(x, attention_mask=self.attn, return_dict=True)["logits"].float(), -1)
            kl = (lp.exp() * (lp - lq)).sum(-1)                      # (B, T)
            nll = -lq.gather(-1, y.unsqueeze(-1)).squeeze(-1)         # (B, T)
            tot_kl += kl.sum().item(); tot_nll += nll.sum().item(); n += kl.numel()
            g, ng, fr = self.tok.token_type_masks(y.cpu(), input_ids=x.cpu())
            for name, m in (("grammar", g), ("non_grammar", ng), ("fact_recall", fr)):
                m = m.to("cuda")
                c = int(m.sum().item())
                if c:
                    kl_t[name] += kl[m].sum().item(); n_t[name] += c
        out = {"kl_pre_post": tot_kl / n, "sl2_ppl": math.exp(tot_nll / n)}
        for name in kl_t:
            out[f"kl_pre_post_{name}"] = kl_t[name] / n_t[name] if n_t[name] else 0.0
        return out


def layer_drift(eff_sd, pre_sd):
    """Relative L2 change per layer: ||theta_i - theta_i^pre|| / ||theta_i^pre||."""
    num, den = {}, {}
    for k, v in eff_sd.items():
        if k.startswith("hybrid.layers."):
            key = "layer%d" % int(k.split(".")[2])
        elif k.startswith("hybrid.final_layer_norm."):
            key = "final_ln"
        elif k.startswith("hybrid.embed_in."):
            key = "embed_in"
        elif k.startswith("embed_out."):
            key = "embed_out"
        else:
            continue
        if k not in pre_sd or pre_sd[k].shape != v.shape:
            continue
        # pre_sd is on cpu, v on cuda
        pv = pre_sd[k].to(device=v.device, dtype=torch.float32)
        d = (v.float() - pv).pow(2).sum().item()
        p = pv.pow(2).sum().item()
        num[key] = num.get(key, 0.0) + d; den[key] = den.get(key, 0.0) + p
    out = {k: math.sqrt(num[k] / den[k]) if den[k] > 0 else 0.0 for k in num}
    bb = [k for k in num if k not in ("embed_in", "embed_out")]
    out["backbone"] = math.sqrt(sum(num[k] for k in bb) / sum(den[k] for k in bb)) if bb else 0.0
    return out


# one post-training run
def build_model(config_name, lr, task, variant, pre_model):
    args = task_args(config_name, lr, task)
    tokenizer = get_tokenizer(args)
    if variant in ("shared", "mixed"):
        assert TOK_SL2 is not None
        tokenizer = SharedVocabTokenizer(tokenizer, TOK_SL2)
    train_dataset = get_train_dataset(args, tokenizer)
    model = get_model(args, tokenizer)
    if variant not in NO_PRETRAIN:
        model = transfer_backbone(pre_model, model)   # shared: embeddings transfer too
    pre_eff = effective_state_dict(model) if variant not in NO_PRETRAIN else None
    if variant == "lora":
        n = apply_lora(model, r=LORA_R, alpha=LORA_ALPHA)
        print(f"    LoRA: wrapped {n} linear layers, trainable {count_params(model, True):,} "
              f"/ {count_params(model):,}", flush=True)
    return args, tokenizer, train_dataset, model, pre_eff


def run_epochs(args, model, tokenizer, train_dataset, variant, n_epochs,
               pre_eff=None, probe=None, log_every=None):
    wd = SCRATCH_WD if variant == "scratch" else POSTTRAIN_WD
    optimizer = AdamW(trainable_parameters(model), lr=args.lr, weight_decay=wd)
    accelerator = Accelerator()
    model, optimizer = accelerator.prepare(model, optimizer)
    # same schedule horizon as train_utils.train
    sched = custom_get_scheduler(optimizer, args.epochs * args.num_examples // args.gradient_accumulation_steps)
    attn = torch.ones((args.sequence_length, args.sequence_length))
    attn = (torch.triu(attn, diagonal=0) - torch.triu(attn, diagonal=args.window)).T.to("cuda")

    hist = {"train_accs": [], "train_losses": [], "drift": [], "probe": []}
    unwrapped = accelerator.unwrap_model(model)

    def monitor(epoch):
        if pre_eff is None:
            return
        eff = effective_state_dict(unwrapped)
        d = layer_drift(eff, pre_eff); d["epoch"] = epoch
        hist["drift"].append(d)
        if probe is not None and (epoch % PROBE_EVERY == 0 or epoch == n_epochs):
            if variant in ("shared", "mixed"):
                p = probe.evaluate(eff, include_embeddings=True)      # whole post-trained model
                pb = probe.evaluate(eff, include_embeddings=False)    # backbone-only, as for full/lora
                p["kl_pre_post_backbone_only"] = pb["kl_pre_post"]
                p["sl2_ppl_backbone_only"] = pb["sl2_ppl"]
            else:
                p = probe.evaluate(eff)
            p["epoch"] = epoch
            hist["probe"].append(p)

    monitor(0)
    for epoch in range(n_epochs):
        model.train()
        loss = train_epoch(model, epoch, accelerator, optimizer, sched, attn, train_dataset, args)
        model.eval()
        _, _, char_acc = evaluation(args, model, tokenizer, do_print=False)
        hist["train_accs"].append(float(char_acc[0])); hist["train_losses"].append(float(loss))
        monitor(epoch + 1)
        if log_every and ((epoch + 1) % log_every == 0 or epoch == n_epochs - 1):
            extra = ""
            if hist["drift"]:
                extra += f" drift={hist['drift'][-1]['backbone']:.4f}"
            if hist["probe"]:
                extra += f" KL(pre||post)={hist['probe'][-1]['kl_pre_post']:.4f} sl2_ppl={hist['probe'][-1]['sl2_ppl']:.2f}"
            print(f"    ep {epoch+1:2d}: acc={char_acc[0]:.4f} loss={loss:.4f}{extra}", flush=True)
    return hist, unwrapped


def find_best_lr(config_name, task, variant, pre_model):
    cache_path = os.path.join(RESULTS_DIR, "auto_lr.json")
    key = f"{task}/{variant}/{config_name}"
    cache = {}
    if os.path.exists(cache_path):
        try:
            cache = json.load(open(cache_path))
        except (json.JSONDecodeError, OSError):
            cache = {}
    if key in cache:
        return cache[key]

    losses = []
    for it in range(LR_ITERS):
        for lr in LR_GRIDS[variant]:
            set_all_seeds(it)
            args, tokenizer, train_dataset, model, _ = build_model(config_name, lr, task, variant, pre_model)
            hist, _ = run_epochs(args, model, tokenizer, train_dataset, variant, LR_SEARCH_EPOCHS)
            losses.append((hist["train_losses"][-1], float(lr)))
            del model, tokenizer, train_dataset
            gc.collect(); torch.cuda.empty_cache()
    losses.sort(key=lambda t: t[0])
    best = losses[0][1]
    cache[key] = best
    os.makedirs(RESULTS_DIR, exist_ok=True)
    json.dump(cache, open(cache_path, "w"), indent=2)
    print(f"  auto-LR {key}: best={best:.2e}  (grid {LR_GRIDS[variant][0]:.0e}..{LR_GRIDS[variant][-1]:.0e})", flush=True)
    return best


def posttrain_single(config_name, seed, task, variant, lr, pretrain_lr,
                     train_sequences, val_sequences, tok_sl2):
    rpath = os.path.join(RESULTS_DIR, task, variant, config_name, f"{config_name}__seed{seed}.json")
    if os.path.exists(rpath):
        print(f"  SKIP {task}/{variant} {config_name} seed={seed} (exists)", flush=True)
        return

    pre_model, probe = None, None
    if variant not in NO_PRETRAIN:
        set_all_seeds(seed)
        pre_model, _ = get_pretrained(config_name, pretrain_lr, seed, variant, train_sequences, val_sequences, tok_sl2)
        probe = SL2Probe(config_name, pretrain_lr, pre_model, val_sequences, tok_sl2)

    print(f"\n[{task}/{variant} {config_name} seed={seed}] lr={lr:.2e}", flush=True)
    set_all_seeds(seed)
    args, tokenizer, train_dataset, model, pre_eff = build_model(config_name, lr, task, variant, pre_model)
    n_total, n_train = count_params(model), count_params(model, True)
    acc_before = None
    if variant not in NO_PRETRAIN:  # accuracy before post-training
        model.eval()
        _, _, acc0 = evaluation(args, model.to("cuda"), tokenizer, do_print=False)
        acc_before = float(np.mean(acc0))
        print(f"    accuracy before post-training: {acc_before:.4f}", flush=True)

    t0 = time.time()
    hist, model = run_epochs(args, model, tokenizer, train_dataset, variant, args.epochs,
                             pre_eff=pre_eff, probe=probe, log_every=8)
    model.eval()
    str_mean, _, char_acc = evaluation(args, model, tokenizer, do_print=False)
    wall = time.time() - t0

    ck = os.path.join(CKPT_DIR, task, variant, config_name, f"seed{seed}.pt")
    os.makedirs(os.path.dirname(ck), exist_ok=True)
    torch.save({k: v.cpu() for k, v in effective_state_dict(model, include_buffers=True).items()}, ck)

    result = {
        "config": config_name, "architecture": RE.CONFIGS[config_name], "seed": seed,
        "task": task, "variant": variant, "pretrain_source": None if variant in NO_PRETRAIN else ("sl2+tasks" if variant == "mixed" else "sl2"),
        "pretrain_lr": None if variant in NO_PRETRAIN else pretrain_lr, "posttrain_lr": lr,
        "weight_decay": SCRATCH_WD if variant == "scratch" else POSTTRAIN_WD,
        "lora": {"r": LORA_R, "alpha": LORA_ALPHA} if variant == "lora" else None,
        "vocab": "sl2_shared" if variant in ("shared", "mixed") else "task",
        "pretrain_corpus": ("sl2+tasks(%.0f%%)" % (100 * MIX_FRAC)) if variant == "mixed" else ("sl2" if variant not in NO_PRETRAIN else None),
        "acc_before_posttrain": acc_before,
        "token_map": getattr(tokenizer, "token_map", None),
        "posttrain_epochs": args.epochs, "hidden": V1.PRETRAIN_HIDDEN, "window": V1.POSTTRAIN_WINDOW,
        "params_total": n_total, "params_trainable": n_train,
        "wall_time_s": wall,
        "train_accs": hist["train_accs"], "train_losses": hist["train_losses"],
        "drift": hist["drift"], "probe": hist["probe"],
        "final_acc": [float(a) for a in char_acc], "mean_acc": float(np.mean(char_acc)),
        "str_acc": [float(s) for s in str_mean], "final_loss": hist["train_losses"][-1],
        "checkpoint": os.path.relpath(ck, RESULTS_DIR),
    }
    os.makedirs(os.path.dirname(rpath), exist_ok=True)
    json.dump(result, open(rpath, "w"), indent=2)
    msg = f"  -> mean_acc={result['mean_acc']:.4f} best_acc={max(hist['train_accs']):.4f} ({wall:.0f}s)"
    if hist["drift"]:
        msg += f" final_drift={hist['drift'][-1]['backbone']:.4f}"
    if hist["probe"]:
        msg += f" KL(pre||post)={hist['probe'][-1]['kl_pre_post']:.4f}"
    print(msg, flush=True)

    del model, tokenizer, train_dataset, pre_model, probe
    gc.collect(); torch.cuda.empty_cache()


# main
def parse_list(tokens, default):
    """Accept `--tasks mkar,mqar`, `--tasks mkar mqar`, or nothing (-> default)."""
    out = [x for t in (tokens or []) for x in t.split(",") if x]
    return out or default


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="*", default=[], help="default: " + " ".join(TASKS))
    ap.add_argument("--variants", nargs="*", default=[], help="from full lora scratch scratch_matched shared mixed; default full lora")
    ap.add_argument("--configs", nargs="*", default=[])
    ap.add_argument("--seeds", nargs="*", default=[])
    ap.add_argument("--pretrain_only", action="store_true")
    ap.add_argument("--epochs", type=int, default=0, help="override post-train epochs (smoke tests)")
    ap.add_argument("--results_dir", default="", help="override results dir (smoke tests)")
    a = ap.parse_args()
    if a.epochs:
        V1.POSTTRAIN_EPOCHS = a.epochs
    if a.results_dir:
        RESULTS_DIR = os.path.abspath(a.results_dir)
        CKPT_DIR = os.path.join(RESULTS_DIR, "checkpoints")
    tasks = parse_list(a.tasks, TASKS)
    variants = parse_list(a.variants, VARIANTS)
    configs = parse_list(a.configs, SUBSET)
    seeds = [int(s) for s in parse_list(a.seeds, [str(s) for s in SEEDS])]

    print("=== Post-training v2: SL2 -> " + "/".join(tasks) + " ===", flush=True)
    print(f"variants={variants} configs={configs} seeds={seeds}", flush=True)
    print(f"full: wd={POSTTRAIN_WD} LR grid {LR_GRIDS['full'][0]:.0e}..{LR_GRIDS['full'][-1]:.0e} | "
          f"lora: r={LORA_R} alpha={LORA_ALPHA} wd={POSTTRAIN_WD} LR grid {LR_GRIDS['lora'][0]:.0e}..{LR_GRIDS['lora'][-1]:.0e} | "
          f"scratch: wd={SCRATCH_WD}", flush=True)

    sl = load_language()
    tok_sl2 = SL2Tokenizer(sl)
    TOK_SL2 = tok_sl2
    train_sequences = generate_corpus(sl, tok_sl2, V1.N_TRAIN_SENTENCES, V1.SENTENCE_LEN, seed=999)
    val_sequences = generate_corpus(sl, tok_sl2, V1.N_VAL_SENTENCES, V1.SENTENCE_LEN, seed=888)
    print(f"SL2 vocab={len(tok_sl2)} train={sum(map(len, train_sequences)):,} tokens", flush=True)

    sl2_lr_cache = os.path.join(_HERE, "..", "..", "results", "synthetic", "lang_sl2", "auto_lr.json")
    pretrain_lrs = json.load(open(sl2_lr_cache)) if os.path.exists(sl2_lr_cache) else {}
    for c in configs:
        pretrain_lrs.setdefault(c, 3e-4)

    need_pretrain = any(v not in NO_PRETRAIN + ("mixed",) for v in variants)
    if need_pretrain:
        print("\nStep 1: pre-train (or load) all config x seed checkpoints", flush=True)
        for c in configs:
            for s in seeds:
                set_all_seeds(s)
                m, _ = pretrain_model(c, pretrain_lrs[c], s, train_sequences, val_sequences, tok_sl2)
                del m; gc.collect(); torch.cuda.empty_cache()
    if "mixed" in variants:
        print(f"\nStep 1b: pre-train (or load) the MIXED checkpoints (SL2 + {100*MIX_FRAC:.0f}% task sequences)", flush=True)
        for c in configs:
            for s in seeds:
                set_all_seeds(s)
                m, _ = pretrain_model_mixed(c, pretrain_lrs[c], s, train_sequences, val_sequences, tok_sl2)
                del m; gc.collect(); torch.cuda.empty_cache()
    if a.pretrain_only:
        sys.exit(0)

    for task in tasks:
        for variant in variants:
            print(f"\n{'='*60}\n=== {task} / {variant} ===\n{'='*60}", flush=True)
            lrs = {}
            for c in configs:
                pre0 = None
                if variant not in NO_PRETRAIN:
                    set_all_seeds(0)
                    pre0, _ = get_pretrained(c, pretrain_lrs[c], 0, variant, train_sequences, val_sequences, tok_sl2)
                lrs[c] = find_best_lr(c, task, variant, pre0)
                del pre0; gc.collect(); torch.cuda.empty_cache()
            for c in configs:
                for s in seeds:
                    posttrain_single(c, s, task, variant, lrs[c], pretrain_lrs[c],
                                     train_sequences, val_sequences, tok_sl2)
    print("\n=== Post-training v2 complete ===", flush=True)
