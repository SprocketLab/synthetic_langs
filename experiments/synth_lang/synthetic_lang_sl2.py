#!/usr/bin/env python3
"""Train the 8-layer hybrids on a synthetic language (SL2 or sl-eng) and evaluate them.

Defaults: d=256, window 100, sequence length 200, batch 32, 15 epochs, 20k training sentences,
early stopping (patience 5), per-configuration auto-LR (2-epoch search over a geometric grid).
The paper uses --window 20 throughout.

Metrics per epoch: validation perplexity and loss, KL(P_true || P_model) decomposed by word type
(type, within grammar, within non-grammar), and recall accuracy (per token and per sentence).

    python synthetic_lang_sl2.py --window 20                          # SL2 (langs/sl2) -> lang_sl2_w20/
    python synthetic_lang_sl2.py --lang ng5 --window 20               # a point of the sl2_sweep.SWEEP grid
    python synthetic_lang_sl2.py --lang my --set N_GRAMMAR=15 --set FACT_DENSITY=0.02   # ad-hoc language
    python synthetic_lang_sl2.py --lang-dir sl-eng-d04 --window 20 --patience 0 \
        --lr-max 3e-3 --lr-points 10                                  # sl-eng (langs/sl-eng-d04)

Results go to results/synthetic/lang_sl2[_w<window>]/ or results/synthetic/lang_sl2_sweep/<lang>[_w<window>]/.
A generated language is saved in <results>/language/ so each run is reproducible.
Other flags: --configs/--seeds (subset), --epochs/--n-train/--n-val, --lr CONFIG=LR (skip the search),
--no-ckpt, --lang-only.
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
from train_utils import ce_loss, custom_get_scheduler, train_epoch

_LANGS_DIR = os.path.join(_HERE, "..", "..", "langs")
sys.path.insert(0, _LANGS_DIR)
from sl2 import SL2, SL2_Config

# hyperparameters

HIDDEN_SIZE = 256
WINDOW = 100
SEQ_LEN = 200
BATCH_SIZE = 32
EPOCHS = 15
EARLY_STOP_PATIENCE = 5
SAVE_CKPT = True         # final weights (fp16) -> <results>/checkpoints/<config>/seed<k>.pt
HEADS = 1
STATE_DIM = 1

N_TRAIN_SENTENCES = 20000
N_VAL_SENTENCES = 2000
SENTENCE_LEN = 256

LR_GRID = list(np.geomspace(1e-4, 1e-3, 7))
LR_ITERS = 2
LR_SEARCH_EPOCHS = 2
FIXED_LR = {}             # --lr: config -> learning rate, skips the auto-LR search

SUBSET = ["pure_ssm", "pure_tf", "4s4t_alternate", "6s2t_start", "6s2t_even", "7s1t_mid",
          "pure_dp", "4d4t_alternate", "6d2t_start"]
SEEDS = [0, 1, 2, 3, 4]

RESULTS_DIR = os.path.abspath(os.path.join(_HERE, "..", "..", "results", "synthetic", "lang_sl2"))
SWEEP_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", "results", "synthetic", "lang_sl2_sweep"))

# language selection, set by apply_args
LANG_NAME = None          # sweep-point name or ad-hoc name; None -> saved langs/sl2
LANG_DIR = None           # None -> langs/sl2
LANG_OVERRIDES = {}       # SL2_Config overrides used to generate it
LANG_SEED = 0


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# language

def make_language(overrides=None, seed=None, dirname=None):
    """Generate an SL2 language from SL2_Config defaults + `overrides` with a fixed seed and save it
    to `dirname` (absolute; default langs/sl2). Runs from langs/ so aux/words_alpha.txt resolves."""
    overrides = dict(overrides if overrides is not None else LANG_OVERRIDES)
    seed = LANG_SEED if seed is None else seed
    unknown = set(overrides) - set(vars(SL2_Config()))
    if unknown:
        raise ValueError(f"unknown SL2_Config field(s): {sorted(unknown)}")
    _prev = os.getcwd()
    os.chdir(_LANGS_DIR)
    try:
        random.seed(seed); np.random.seed(seed)
        config = SL2_Config(**overrides)
        sl = SL2(config=config)
        if dirname:
            os.makedirs(os.path.dirname(os.path.abspath(dirname)), exist_ok=True)
        sl.save(dirname=dirname)          # SL2.save joins with cwd; absolute dirname wins
    finally:
        os.chdir(_prev)
    return sl


def load_language(dirname=None):
    """Load the language from `dirname` (default LANG_DIR, then langs/sl2); generate it from
    LANG_OVERRIDES / LANG_SEED if missing."""
    dirname = dirname or LANG_DIR
    _prev = os.getcwd()
    os.chdir(_LANGS_DIR)
    try:
        sl = SL2(load=True, dirname=dirname)
    except (AssertionError, FileNotFoundError):
        import sl2_sweep, shutil
        src = sl2_sweep.COPY_FROM.get(LANG_NAME)
        if dirname and src and os.path.isdir(os.path.join(_LANGS_DIR, src)):
            print(f"No saved SL2 at {dirname}, copying langs/{src} there...", flush=True)
            shutil.copytree(os.path.join(_LANGS_DIR, src), dirname)
            sl = SL2(load=True, dirname=dirname)
        else:
            print(f"No saved SL2 at {dirname or 'langs/sl2'}, generating (seed={LANG_SEED}, "
                  f"overrides={LANG_OVERRIDES})...", flush=True)
            sl = make_language(dirname=dirname)
    finally:
        os.chdir(_prev)
    return sl


def save_language_config(sl):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "language_config.json"), "w") as f:
        json.dump({
            "N_GRAMMAR": sl.config.N_GRAMMAR,
            "N_VOCAB": sl.config.N_VOCAB,
            "N_FACT_MARK": sl.config.N_FACT_MARK,
            "MULTI_FACT": sl.config.MULTI_FACT,
            "USE_RECALL": sl.config.USE_RECALL,
            "USE_PRIOR": sl.config.USE_PRIOR,
            "FACT_DENSITY": sl.config.FACT_DENSITY,
            "RECALL_DENSITY": sl.config.RECALL_DENSITY,
            "OTHER_DENSITY": sl.config.OTHER_DENSITY,
            "NON_GRAM_TYPE": sl.config.NON_GRAM_TYPE,
            "vocab_size": len(sl.config.all_vocab),
            "lang": LANG_NAME,
            "lang_dir": LANG_DIR,
            "lang_overrides": LANG_OVERRIDES,
            "lang_seed": LANG_SEED,
        }, f, indent=2)


# tokenizer

class WordTokenizer:
    def __init__(self, sl):
        self.word2id = dict(sl.vtoi)
        self.id2word = list(sl.config.all_vocab)
        self.vocab_size = len(self.id2word)

        grammar_set = set(sl.config.grammar)
        non_grammar_set = set(sl.config.non_grammar)
        self.grammar_ids = set(self.word2id[w] for w in grammar_set if w in self.word2id)
        self.non_grammar_ids = set(self.word2id[w] for w in non_grammar_set if w in self.word2id)
        markers = set()
        for i in range(1, sl.config.N_FACT_MARK + 1):
            for prefix in ("fact", "recall"):
                w = f"{prefix}{i}"
                if w in self.word2id:
                    markers.add(w)
        self.fact_recall_ids = set(self.word2id[w] for w in markers)
        self.fact_ids = set(self.word2id[w] for w in markers if w.startswith("fact"))
        self.recall_ids = set(self.word2id[w] for w in markers if w.startswith("recall"))
        self.type_index = {                      # partition of the vocab by next-token type
            "grammar": torch.tensor(sorted(self.grammar_ids), dtype=torch.long),
            "non_grammar": torch.tensor(sorted(self.non_grammar_ids), dtype=torch.long),
            "marker": torch.tensor(sorted(self.fact_recall_ids), dtype=torch.long),
        }
        self._fact_t = torch.tensor(sorted(self.fact_ids), dtype=torch.long)
        self._recall_t = torch.tensor(sorted(self.recall_ids), dtype=torch.long)
        self.use_recall = bool(sl.config.USE_RECALL)
        self.start_id = self.word2id.get("<start>")

    def marker_kind_masks(self, prev_ids, before=()):
        """(after_fact, after_recall) masks: positions whose preceding token is a fact# (definition)
        or recall# (recall) marker. Their union is the fact_recall mask of token_type_masks.
        Without recall# tokens (USE_RECALL=False, sl-eng) the first fact# of a number in a sentence
        is the definition and a repeat is the recall. `before` holds the tokens preceding
        prev_ids[0]; <start> resets. prev_ids is (seq,) or (batch, seq), rows scanned independently."""
        is_fact = torch.isin(prev_ids, self._fact_t)
        if self.use_recall:
            return is_fact, torch.isin(prev_ids, self._recall_t)
        rows = prev_ids.reshape(-1, prev_ids.shape[-1]).tolist()
        again = torch.zeros(len(rows), prev_ids.shape[-1], dtype=torch.bool)
        for r, row in enumerate(rows):
            seen = set(t for t in before if t in self.fact_ids) if prev_ids.dim() == 1 else set()
            for j, t in enumerate(row):
                if t == self.start_id:
                    seen = set()
                elif t in self.fact_ids:
                    again[r, j] = t in seen
                    seen.add(t)
        again = again.reshape(prev_ids.shape).to(prev_ids.device)
        return is_fact & ~again, again

    def token_type_masks(self, token_ids, input_ids=None):
        """Boolean masks for grammar, non-grammar and fact-value targets.

        fact_recall marks the values right after a fact#/recall# token. The markers themselves
        are excluded from all three masks.

        Args:
            token_ids: target ids, (batch, seq) or (seq,)
            input_ids: matching input ids; the preceding token is read from here if given,
                       otherwise from token_ids shifted by one.
        """
        grammar = torch.zeros_like(token_ids, dtype=torch.bool)
        non_grammar = torch.zeros_like(token_ids, dtype=torch.bool)
        for gid in self.grammar_ids:
            grammar |= (token_ids == gid)
        for nid in self.non_grammar_ids:
            non_grammar |= (token_ids == nid)

        if input_ids is not None:
            prev = input_ids
        else:
            prev = token_ids

        is_marker = torch.zeros_like(prev, dtype=torch.bool)
        for fid in self.fact_recall_ids:
            is_marker |= (prev == fid)

        fact_recall = torch.zeros_like(token_ids, dtype=torch.bool)
        if input_ids is not None:
            fact_recall = is_marker
        else:
            fact_recall[..., 1:] = is_marker[..., :-1]

        marker_on_target = torch.zeros_like(token_ids, dtype=torch.bool)
        for fid in self.fact_recall_ids:
            marker_on_target |= (token_ids == fid)
        grammar &= ~marker_on_target
        non_grammar &= ~marker_on_target
        non_grammar &= ~fact_recall

        return grammar, non_grammar, fact_recall

    def encode(self, text):
        return [self.word2id[w] for w in text.split() if w in self.word2id]

    def __len__(self):
        return self.vocab_size


# dataset

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


# corpus generation

def generate_corpus(sl, tok, n_sentences, sentence_len, seed=42, with_probs=False):
    rng_state = random.getstate()
    np_state = np.random.get_state()
    random.seed(seed)
    np.random.seed(seed)

    sequences = []
    all_probs = [] if with_probs else None

    for _ in range(n_sentences):
        if with_probs:
            text, probs = sl.generate_sentence(sentence_len, get_probs=True)
            all_probs.append(probs)
        else:
            text = sl.generate_sentence(sentence_len)
        ids = tok.encode(text)
        sequences.append(ids)

    random.setstate(rng_state)
    np.random.set_state(np_state)
    return sequences, all_probs


def find_recall_positions(sequences, tok, sl):
    """Find recall positions. With USE_RECALL=True: recall# tokens trigger recall."""
    recall_token_to_num = {}
    for i in range(1, sl.config.N_FACT_MARK + 1):
        if sl.config.USE_RECALL:
            rid = tok.word2id.get(f"recall{i}")
        else:
            rid = tok.word2id.get(f"fact{i}")
        if rid is not None:
            recall_token_to_num[rid] = i

    fact_token_to_num = {}
    for i in range(1, sl.config.N_FACT_MARK + 1):
        fid = tok.word2id.get(f"fact{i}")
        if fid is not None:
            fact_token_to_num[fid] = i

    positions = []
    for si, seq in enumerate(sequences):
        defined = {}
        for pos in range(len(seq) - 1):
            if sl.config.USE_RECALL:
                if seq[pos] in fact_token_to_num:
                    defined[fact_token_to_num[seq[pos]]] = seq[pos + 1]
                elif seq[pos] in recall_token_to_num:
                    fnum = recall_token_to_num[seq[pos]]
                    if fnum in defined:
                        positions.append((si, pos + 1, seq[pos + 1]))
            elif seq[pos] in fact_token_to_num:
                # without recall# tokens the first fact# is the definition (not scored)
                # and a repeat is the recall
                fnum = fact_token_to_num[seq[pos]]
                if fnum in defined:
                    positions.append((si, pos + 1, defined[fnum]))
                else:
                    defined[fnum] = seq[pos + 1]
    return positions


# helpers

def result_path(config_name, lr, seed):
    lr_tag = f"lr{lr:.0e}"
    return os.path.join(RESULTS_DIR, config_name,
                        f"{config_name}__{lr_tag}__seed{seed}.json")


def make_args(config_name, lr):
    layers = RE.layers_list(config_name)
    return Namespace(
        nope=False, layers=layers,
        hidden_size=HIDDEN_SIZE, heads=HEADS, state_dim=STATE_DIM,
        lr=lr, epochs=EPOCHS,
        sequence_length=SEQ_LEN, window=WINDOW,
        gradient_accumulation_steps=1, progress_bar=False,
        num_examples=999_999, num_log_steps=50,
    )


# auto-LR

def auto_lr_path():
    return os.path.join(RESULTS_DIR, "auto_lr.json")


def load_auto_lr():
    p = auto_lr_path()
    if os.path.exists(p):
        try:
            with open(p) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def save_auto_lr(cache):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(auto_lr_path(), "w") as f:
        json.dump(cache, f, indent=2)


def find_best_lr(config_name, train_sequences, tok, force=False):
    cache = load_auto_lr()
    if config_name in cache and not force:
        return cache[config_name]

    losses = []
    for it in range(LR_ITERS):
        for lr in LR_GRID:
            set_all_seeds(it)
            args = make_args(config_name, lr)
            model = get_model(args, tok)
            train_ds = LangDataset(train_sequences, SEQ_LEN, BATCH_SIZE, shuffle=True)

            optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.1)
            accelerator = Accelerator()
            model, optimizer = accelerator.prepare(model, optimizer)
            lr_scheduler = custom_get_scheduler(optimizer, len(train_ds) * EPOCHS)

            attn = torch.ones((SEQ_LEN, SEQ_LEN))
            attn = (torch.triu(attn, 0) - torch.triu(attn, WINDOW)).T.to("cuda")

            model.train()
            for ep in range(LR_SEARCH_EPOCHS):
                avg_loss = train_epoch(model, ep, accelerator, optimizer,
                                       lr_scheduler, attn, train_ds, args)
            losses.append((float(avg_loss), float(lr)))

            del model, optimizer, accelerator, lr_scheduler, train_ds
            gc.collect(); torch.cuda.empty_cache()

    losses.sort(key=lambda t: t[0])
    best = losses[0][1]
    cache[config_name] = best
    save_auto_lr(cache)
    print(f"  auto-LR {config_name:16s} ({RE.CONFIGS[config_name]}): best={best:.2e} "
          f"(min {LR_SEARCH_EPOCHS}-ep loss {losses[0][0]:.4f})", flush=True)
    return best


# evaluation

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
    avg_loss = total_loss / total_tokens
    return math.exp(avg_loss), avg_loss


def eval_sentences(model, val_sequences, val_probs, tok, sl, attn_mask):
    """Per-sentence evaluation: each sentence is fed from its <start> and scored on
    seq[3 : input_len + 1], the positions used by eval_kl_divergence.
    eval_recall_accuracy / eval_perplexity chunk the concatenated stream instead, which cuts
    some bindings off from their recalls (recall ceiling 0.74 instead of 0.94 on SL2).
    Returns ppl (mean CE over scored positions), recall accuracy over recall positions inside
    the input, and its Bayes ceiling (mean max_v P_true(v); < 1 when MULTI_FACT binds several
    values to one fact#)."""
    model.eval()
    dev = next(model.parameters()).device
    by_seq = {}
    for si, pos, target in find_recall_positions(val_sequences, tok, sl):
        by_seq.setdefault(si, []).append((pos, target))
    nll, n_tok = 0.0, 0
    correct, n_rec, ceiling = 0, 0, 0.0
    with torch.no_grad():
        for si0 in range(0, len(val_sequences), BATCH_SIZE):
            batch_seqs = val_sequences[si0:si0 + BATCH_SIZE]
            input_len = min(min(len(s) for s in batch_seqs) - 1, SEQ_LEN)
            x = torch.stack([torch.tensor(s[:input_len], dtype=torch.long) for s in batch_seqs]).to(dev)
            logits = model(x, attention_mask=attn_mask[:input_len, :input_len], return_dict=True)["logits"]
            log_q = F.log_softmax(logits.float(), dim=-1)
            y = torch.stack([torch.tensor(s[3:input_len + 1], dtype=torch.long) for s in batch_seqs]).to(dev)
            nll -= log_q[:, 2:input_len].gather(-1, y.unsqueeze(-1)).sum().item()
            n_tok += y.numel()
            pred = log_q.argmax(-1).cpu()
            for i in range(len(batch_seqs)):
                for pos, target in by_seq.get(si0 + i, []):
                    if 3 <= pos <= input_len:            # target seq[pos] is predicted at logits[pos - 1]
                        n_rec += 1
                        correct += int(pred[i, pos - 1].item() == target)
                        ceiling += float(np.max(val_probs[si0 + i][pos - 3]))
    return {"ppl": math.exp(nll / n_tok) if n_tok else float("nan"),
            "recall_acc": correct / n_rec if n_rec else 0.0,
            "recall_n": n_rec,
            "recall_ceiling": ceiling / n_rec if n_rec else 0.0}


def eval_recall_accuracy(model, val_sequences, tok, sl, attn_mask):
    model.eval()
    flat = []
    for s in val_sequences:
        flat.extend(s)
    flat_t = torch.tensor(flat, dtype=torch.long)

    n_chunks = len(flat_t) // (SEQ_LEN + 1)
    flat_t = flat_t[:n_chunks * (SEQ_LEN + 1)]
    chunks = flat_t.view(n_chunks, SEQ_LEN + 1)

    all_preds = []
    with torch.no_grad():
        for ci in range(0, len(chunks), 32):
            batch_in = chunks[ci:ci + 32, :-1].to("cuda")
            logits = model(batch_in, attention_mask=attn_mask, return_dict=True)["logits"]
            all_preds.append(logits.argmax(dim=-1).cpu())
    all_preds = torch.cat(all_preds, dim=0)

    seq_offsets, offset = [], 0
    for s in val_sequences:
        seq_offsets.append(offset)
        offset += len(s)

    recall_positions = find_recall_positions(val_sequences, tok, sl)
    total_usable = n_chunks * (SEQ_LEN + 1)
    correct, total = 0, 0
    for si, pos, target_id in recall_positions:
        gpos = seq_offsets[si] + pos
        if gpos >= total_usable:
            continue
        ci = gpos // (SEQ_LEN + 1)
        pi = gpos % (SEQ_LEN + 1)
        if pi == 0 or ci >= len(all_preds):
            continue
        total += 1
        if all_preds[ci, pi - 1].item() == target_id:
            correct += 1
    return correct / total if total > 0 else 0.0, total


def eval_kl_divergence(model, val_sequences, val_probs, tok, attn_mask):
    """KL(P_true || P_model) per position, averaged overall and by token type.

    Coarse split by target type:
      grammar / non_grammar  positions whose target is a grammar / non-grammar word. Non-grammar
                             words are inserted at a context-independent rate, so this is close
                             to a random split of the free positions (kl_grammar ~ kl_non_grammar).
      fact_recall            positions right after a fact#/recall# marker (the value).
    Fine split on the free positions (not after a marker), chain rule over the
    next-token type T in {grammar, non_grammar, marker}:
        KL(p||q) = KL(p_T||q_T) + sum_t p_t * KL(p(.|t) || q(.|t))
      free                  total KL on free positions
      type                  KL between true and modelled type distributions
      within_<t>            p_t-weighted mean of KL(p(.|t)||q(.|t))
      p_<t>                 mean true type probability (weights of the identity above)
      fact_def / recall     KL at the value after a fact# (true dist uniform over the 65
                            non-grammar words) / after a recall# (the memory test).
    Stored in the run JSON with a "kl_" prefix (kl_free, kl_type, kl_within_grammar, ...).
    Entropy: mean H(q) of the model and H(p) of the true distribution on the same masks, stored
    as ent_model_<k> / ent_true_<k> for k in grammar, non_grammar, fact_recall, free, fact_def, recall.
    """
    model.eval()
    V = len(tok)
    total_kl = 0.0
    n_positions = 0
    kl_by_type = {"grammar": 0.0, "non_grammar": 0.0, "fact_recall": 0.0}
    n_by_type = {"grammar": 0, "non_grammar": 0, "fact_recall": 0}
    fine = {"free": 0.0, "type": 0.0, "fact_def": 0.0, "recall": 0.0}
    fine_n = {"free": 0, "type": 0, "fact_def": 0, "recall": 0}
    within = {t: 0.0 for t in tok.type_index}   # sum over free positions of p_t * KL_within_t
    pmass = {t: 0.0 for t in tok.type_index}    # sum over free positions of p_t
    ENT_KEYS = ("grammar", "non_grammar", "fact_recall", "free", "fact_def", "recall")
    ent_model = {k: 0.0 for k in ENT_KEYS}; ent_true = {k: 0.0 for k in ENT_KEYS}; ent_n = {k: 0 for k in ENT_KEYS}

    with torch.no_grad():
        for si in range(0, len(val_sequences), BATCH_SIZE):
            batch_seqs = val_sequences[si:si + BATCH_SIZE]
            batch_probs = val_probs[si:si + BATCH_SIZE]
            bsz = len(batch_seqs)

            input_len = min(min(len(s) for s in batch_seqs) - 1, SEQ_LEN)
            input_ids = torch.stack([
                torch.tensor(s[:input_len], dtype=torch.long) for s in batch_seqs
            ]).to("cuda")

            mask = attn_mask[:input_len, :input_len]
            logits = model(input_ids, attention_mask=mask, return_dict=True)["logits"]
            log_q = F.log_softmax(logits, dim=-1).cpu()

            target_ids = torch.stack([
                torch.tensor(batch_seqs[i][3:3 + (input_len - 2)], dtype=torch.long)
                for i in range(bsz)
            ])
            prev_ids = torch.stack([
                torch.tensor(batch_seqs[i][2:2 + (input_len - 2)], dtype=torch.long)
                for i in range(bsz)
            ])
            g_mask, ng_mask, fr_mask = tok.token_type_masks(target_ids, input_ids=prev_ids)

            n_pos = input_len - 2
            for i in range(bsz):
                p_true = torch.from_numpy(
                    batch_probs[i][:n_pos, :V].copy()).float()
                lq = log_q[i, 2:2 + n_pos, :V]

                nonzero = p_true > 1e-10
                log_p = torch.where(nonzero, torch.log(p_true.clamp(min=1e-10)), torch.zeros_like(p_true))
                kl = torch.where(nonzero, p_true * (log_p - lq), torch.zeros_like(p_true))
                kl_per_pos = kl.sum(dim=-1)
                total_kl += kl_per_pos.sum().item()
                n_positions += n_pos
                h_true = -(torch.where(nonzero, p_true * log_p, torch.zeros_like(p_true))).sum(-1)
                h_model = -(lq.exp() * lq).sum(-1)
                after_fact, after_recall = tok.marker_kind_masks(prev_ids[i], before=batch_seqs[i][:2])
                for name, m in (("grammar", g_mask[i]), ("non_grammar", ng_mask[i]), ("fact_recall", fr_mask[i]),
                                ("free", ~fr_mask[i]), ("fact_def", after_fact), ("recall", after_recall)):
                    cnt = int(m.sum().item())
                    if cnt:
                        ent_model[name] += h_model[m].sum().item(); ent_true[name] += h_true[m].sum().item()
                        ent_n[name] += cnt

                for tname, tmask in [("grammar", g_mask[i]),
                                     ("non_grammar", ng_mask[i]),
                                     ("fact_recall", fr_mask[i])]:
                    cnt = tmask.sum().item()
                    if cnt > 0:
                        kl_by_type[tname] += kl_per_pos[tmask].sum().item()
                        n_by_type[tname] += cnt

                # fine split: chain rule over the next-token type on the free positions
                free = ~fr_mask[i]
                nfree = int(free.sum().item())
                if nfree:
                    p_f, kl_f, q_f = p_true[free], kl[free], lq[free].exp()
                    kl_type_pos = torch.zeros(nfree)
                    for t, ix in tok.type_index.items():
                        pt = p_f[:, ix].sum(-1)
                        qt = q_f[:, ix].sum(-1).clamp(min=1e-10)
                        lr = torch.where(pt > 0, torch.log(pt.clamp(min=1e-10)) - torch.log(qt),
                                         torch.zeros_like(pt))
                        kl_type_pos += pt * lr
                        # p_t * KL(p(.|t)||q(.|t)) = sum_{v in t} kl(v) - p_t * (log p_t - log q_t)
                        within[t] += (kl_f[:, ix].sum(-1) - pt * lr).sum().item()
                        pmass[t] += pt.sum().item()
                    fine["free"] += kl_per_pos[free].sum().item(); fine_n["free"] += nfree
                    fine["type"] += kl_type_pos.sum().item();      fine_n["type"] += nfree
                for name, m in (("fact_def", after_fact), ("recall", after_recall)):
                    cnt = int(m.sum().item())
                    if cnt:
                        fine[name] += kl_per_pos[m].sum().item(); fine_n[name] += cnt

    avg_kl = total_kl / n_positions if n_positions > 0 else 0.0
    kl_split = {}
    for t in kl_by_type:
        kl_split[t] = kl_by_type[t] / n_by_type[t] if n_by_type[t] > 0 else 0.0
        kl_split[f"{t}_n"] = n_by_type[t]
    for k in fine:
        kl_split[k] = fine[k] / fine_n[k] if fine_n[k] > 0 else 0.0
        kl_split[f"{k}_n"] = fine_n[k]
    for t in tok.type_index:
        kl_split[f"within_{t}"] = within[t] / pmass[t] if pmass[t] > 0 else 0.0
        kl_split[f"p_{t}"] = pmass[t] / fine_n["free"] if fine_n["free"] > 0 else 0.0
    for k in ENT_KEYS:
        kl_split[f"ent_model_{k}"] = ent_model[k] / ent_n[k] if ent_n[k] else 0.0
        kl_split[f"ent_true_{k}"] = ent_true[k] / ent_n[k] if ent_n[k] else 0.0
    return avg_kl, n_positions, kl_split


# training

def eval_split_loss(model, dataset, attn_mask, tok):
    """Compute loss split by token type: grammar, non-grammar, fact-recall."""
    model.eval()
    totals = {"all": 0.0, "grammar": 0.0, "non_grammar": 0.0, "fact_recall": 0.0,
              "fact_def": 0.0, "recall": 0.0}
    counts = {k: 0 for k in totals}
    with torch.no_grad():
        for i in range(len(dataset)):
            batch = dataset[i]
            x = batch["input_ids"].to("cuda")
            y = batch["output_ids"].to("cuda")
            logits = model(x, attention_mask=attn_mask, return_dict=True)["logits"]
            per_token_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), y.view(-1), reduction="none"
            ).view(y.shape)
            g_mask, ng_mask, fr_mask = tok.token_type_masks(y, input_ids=x)
            fd_mask, rc_mask = tok.marker_kind_masks(x.cpu())      # fact_recall = fd | rc
            fd_mask, rc_mask = fd_mask.to(y.device), rc_mask.to(y.device)
            totals["all"] += per_token_loss.sum().item()
            counts["all"] += y.numel()
            for tname, tmask in [("grammar", g_mask), ("non_grammar", ng_mask),
                                 ("fact_recall", fr_mask), ("fact_def", fd_mask), ("recall", rc_mask)]:
                cnt = tmask.sum().item()
                if cnt > 0:
                    totals[tname] += per_token_loss[tmask].sum().item()
                    counts[tname] += cnt
    result = {}
    for k in totals:
        result[k] = totals[k] / counts[k] if counts[k] > 0 else 0.0
        result[f"{k}_n"] = counts[k]
    return result


def train_and_eval(args, model, train_ds, val_ds, val_sequences, val_probs, tok, sl):
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    accelerator = Accelerator()
    model, optimizer = accelerator.prepare(model, optimizer)

    lr_scheduler = custom_get_scheduler(optimizer, len(train_ds) * args.epochs)

    attn = torch.ones((SEQ_LEN, SEQ_LEN))
    attn = (torch.triu(attn, 0) - torch.triu(attn, WINDOW)).T.to("cuda")

    train_losses, val_ppls, val_losses = [], [], []
    recall_accs, kl_divs, split_losses = [], [], []
    best_ppl = float("inf")
    wait = 0

    for epoch in range(args.epochs):
        model.train()
        avg_loss = train_epoch(model, epoch, accelerator, optimizer,
                               lr_scheduler, attn, train_ds, args)
        train_losses.append(avg_loss)

        ppl, vloss = eval_perplexity(model, val_ds, attn)
        val_ppls.append(ppl)
        val_losses.append(vloss)

        if ppl < best_ppl - 0.01:
            best_ppl = ppl
            wait = 0
        else:
            wait += 1

        stopping = bool(EARLY_STOP_PATIENCE) and wait >= EARLY_STOP_PATIENCE
        if (epoch + 1) % 3 == 0 or epoch == args.epochs - 1 or stopping:
            racc, n_r = eval_recall_accuracy(model, val_sequences, tok, sl, attn)
            sent = eval_sentences(model, val_sequences, val_probs, tok, sl, attn)
            recall_accs.append({"epoch": epoch, "accuracy": racc, "n_positions": n_r,
                                "accuracy_sent": sent["recall_acc"], "n_positions_sent": sent["recall_n"],
                                "ceiling_sent": sent["recall_ceiling"], "ppl_sent": sent["ppl"]})

            kl, n_kl, kl_split = eval_kl_divergence(model, val_sequences, val_probs, tok, attn)
            kl_divs.append({"epoch": epoch, "kl_divergence": kl, "n_positions": n_kl,
                            **{(k if k.startswith("ent_") else f"kl_{k}"): v for k, v in kl_split.items()}})

            sl_loss = eval_split_loss(model, val_ds, attn, tok)
            split_losses.append({"epoch": epoch, **sl_loss})

            print(f"  ep {epoch:2d}: loss={avg_loss:.4f}  ppl={ppl:.1f}  "
                  f"recall={racc:.3f}({n_r}) sent={sent['recall_acc']:.3f}/{sent['recall_ceiling']:.3f}  kl={kl:.3f} "
                  f"[g={kl_split['grammar']:.3f} ng={kl_split['non_grammar']:.3f} "
                  f"fr={kl_split['fact_recall']:.3f}]  "
                  f"loss_g={sl_loss['grammar']:.4f} loss_ng={sl_loss['non_grammar']:.4f} "
                  f"loss_fr={sl_loss['fact_recall']:.4f}",
                  flush=True)
        else:
            print(f"  ep {epoch:2d}: loss={avg_loss:.4f}  ppl={ppl:.1f}", flush=True)

        if stopping:
            print(f"  early stop at epoch {epoch} (val PPL no improvement for {wait} epochs)",
                  flush=True)
            break

    return train_losses, val_ppls, val_losses, recall_accs, kl_divs, split_losses, best_ppl


def run_single(config_name, lr, seed, tok, sl, train_seqs, val_seqs, val_probs):
    rpath = result_path(config_name, lr, seed)
    if os.path.exists(rpath):
        print(f"  SKIP {config_name} seed={seed} (exists)", flush=True)
        return

    set_all_seeds(seed)
    train_ds = LangDataset(train_seqs, SEQ_LEN, BATCH_SIZE, shuffle=True)
    val_ds = LangDataset(val_seqs, SEQ_LEN, BATCH_SIZE, shuffle=False)

    args = make_args(config_name, lr)
    model = get_model(args, tok)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  {config_name} seed={seed} lr={lr:.2e}: {n_params:,} params", flush=True)

    t0 = time.time()
    train_losses, val_ppls, val_losses, recall_accs, kl_divs, split_losses, best_ppl = \
        train_and_eval(args, model, train_ds, val_ds, val_seqs, val_probs, tok, sl)
    wall = time.time() - t0
    actual_epochs = len(train_losses)

    ckpt = None
    if SAVE_CKPT:
        ckpt = os.path.join(RESULTS_DIR, "checkpoints", config_name, f"seed{seed}.pt")
        os.makedirs(os.path.dirname(ckpt), exist_ok=True)
        # params in fp16, buffers unchanged (masked_bias = -1e9 overflows fp16); the causal-mask
        # buffers (attention.bias) are skipped, so load with strict=False and cast with .float()
        pnames = {k for k, _ in model.named_parameters()}
        torch.save({k: (v.half() if k in pnames else v).cpu() for k, v in model.state_dict().items()
                    if not k.endswith("attention.bias")}, ckpt)
        print(f"  saved {ckpt}", flush=True)

    os.makedirs(os.path.dirname(rpath), exist_ok=True)
    with open(rpath, "w") as f:
        json.dump({
            "config": config_name,
            "layers": RE.CONFIGS[config_name],
            "seed": seed,
            "hidden_size": HIDDEN_SIZE,
            "window": WINDOW,
            "seq_len": SEQ_LEN,
            "batch_size": BATCH_SIZE,
            "lr": lr,
            "epochs": EPOCHS,
            "actual_epochs": actual_epochs,
            "early_stopped": actual_epochs < EPOCHS,
            "n_params": n_params,
            "vocab_size": len(tok),
            "multi_fact": sl.config.MULTI_FACT,
            "use_recall": sl.config.USE_RECALL,
            "n_grammar": sl.config.N_GRAMMAR,
            "n_non_grammar": len(sl.config.non_grammar),
            "n_fact_mark": sl.config.N_FACT_MARK,
            "fact_density": sl.config.FACT_DENSITY,
            "recall_density": sl.config.RECALL_DENSITY,
            "other_density": sl.config.OTHER_DENSITY,
            "lang": LANG_NAME,
            "lang_overrides": LANG_OVERRIDES,
            "sentence_len": SENTENCE_LEN,
            "n_train_sentences": N_TRAIN_SENTENCES,
            "n_val_sentences": N_VAL_SENTENCES,
            "train_losses": train_losses,
            "val_ppls": val_ppls,
            "val_losses": val_losses,
            "best_ppl": best_ppl,
            "final_ppl": val_ppls[-1],
            "recall_accs": recall_accs,
            "kl_divs": kl_divs,
            "split_losses": split_losses,
            "final_recall_acc": recall_accs[-1]["accuracy"] if recall_accs else None,
            # per-sentence recall accuracy, its Bayes ceiling, and PPL
            "final_recall_acc_sent": recall_accs[-1]["accuracy_sent"] if recall_accs else None,
            "recall_ceiling_sent": recall_accs[-1]["ceiling_sent"] if recall_accs else None,
            "final_ppl_sent": recall_accs[-1]["ppl_sent"] if recall_accs else None,
            "final_kl_div": kl_divs[-1]["kl_divergence"] if kl_divs else None,
            "wall_time_s": wall,
            "checkpoint": os.path.relpath(ckpt, RESULTS_DIR) if ckpt else None,
        }, f, indent=2)
    print(f"  -> saved {rpath} ({wall:.0f}s)", flush=True)

    del model; gc.collect(); torch.cuda.empty_cache()


# CLI

def _parse_value(v):
    """'0.02' -> 0.02, '20' -> 20, 'true' -> True, anything else -> str."""
    try:
        return json.loads(v)
    except ValueError:
        return {"true": True, "false": False}.get(v.lower(), v)


def build_parser():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_argument_group("language")
    g.add_argument("--lang", default=None,
                   help="language name: a point in sl2_sweep.SWEEP (overrides taken from there) or an "
                        "ad-hoc name used with --set. Default: the saved langs/sl2 (original behaviour).")
    g.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="SL2_Config override (repeatable), e.g. --set N_GRAMMAR=20 --set FACT_DENSITY=0.02")
    g.add_argument("--lang-dir", default=None,
                   help="load a pre-built language from this directory (absolute or relative to langs/). "
                        "Overrides --lang/--set generation; the directory must contain config.json, mc.npy, geom.npy.")
    g.add_argument("--lang-seed", type=int, default=LANG_SEED, help="seed for generating the language")
    g.add_argument("--lang-only", action="store_true", help="generate/inspect the language, no training")
    g = ap.add_argument_group("run")
    g.add_argument("--results-dir", default=None,
                   help="default results/synthetic/lang_sl2 (no --lang) or results/synthetic/lang_sl2_sweep/<lang>")
    g.add_argument("--configs", nargs="+", default=None, help=f"subset of {SUBSET}")
    g.add_argument("--seeds", nargs="+", type=int, default=None)
    g.add_argument("--epochs", type=int, default=None)
    g.add_argument("--n-train", type=int, default=None, help="train sentences (default %d)" % N_TRAIN_SENTENCES)
    g.add_argument("--n-val", type=int, default=None, help="val sentences (default %d)" % N_VAL_SENTENCES)
    g.add_argument("--patience", type=int, default=None,
                   help="early-stop patience in epochs (default %d; 0 disables)" % EARLY_STOP_PATIENCE)
    g.add_argument("--lr-max", type=float, default=None, help="top of the auto-LR grid (default %g)" % LR_GRID[-1])
    g.add_argument("--no-ckpt", action="store_true", help="do not save the final weights")
    g.add_argument("--window", type=int, default=None,
                   help="attention window of the T layers (default %d); a different value writes to <results>_w<window>" % WINDOW)
    g.add_argument("--lr-points", type=int, default=None, help="auto-LR grid size (default %d)" % len(LR_GRID))
    g.add_argument("--lr", nargs="+", default=None, metavar="CONFIG=LR",
                   help="fixed learning rates that skip the auto-LR search (CONFIG=LR pairs or one value for all)")
    return ap


def apply_args(args):
    """Turn parsed args into the module-level settings (RESULTS_DIR, LANG_*, SUBSET, ...)."""
    global RESULTS_DIR, LANG_NAME, LANG_DIR, LANG_OVERRIDES, LANG_SEED
    global SUBSET, SEEDS, EPOCHS, N_TRAIN_SENTENCES, N_VAL_SENTENCES, EARLY_STOP_PATIENCE, LR_GRID, SAVE_CKPT, WINDOW
    global FIXED_LR
    for tok in getattr(args, "lr", None) or []:
        cfg, _, val = tok.rpartition("=")
        FIXED_LR[cfg or "*"] = float(val)
    overrides = {}
    if args.lang is not None:
        import sl2_sweep
        if args.lang in sl2_sweep.SWEEP:
            overrides.update(sl2_sweep.overrides(args.lang))
        elif not args.set:
            raise SystemExit(f"--lang {args.lang!r} is not in sl2_sweep.SWEEP; give --set KEY=VALUE overrides")
    for kv in args.set:
        k, _, v = kv.partition("=")
        if not _:
            raise SystemExit(f"--set expects KEY=VALUE, got {kv!r}")
        overrides[k.strip()] = _parse_value(v.strip())
    if args.set and args.lang is None:
        raise SystemExit("--set needs --lang <name> (the name becomes the results directory)")
    LANG_NAME = args.lang
    LANG_OVERRIDES = overrides
    LANG_SEED = args.lang_seed
    if args.lang_dir:
        d = args.lang_dir
        if not os.path.isabs(d):
            d = os.path.join(_LANGS_DIR, d)
        LANG_DIR = os.path.abspath(d)
        if LANG_NAME is None:
            LANG_NAME = os.path.basename(LANG_DIR)
    if args.results_dir:
        RESULTS_DIR = os.path.abspath(args.results_dir)
    elif args.lang_dir:
        RESULTS_DIR = os.path.join(SWEEP_ROOT, LANG_NAME)
    elif args.lang is not None:
        RESULTS_DIR = os.path.join(SWEEP_ROOT, args.lang)
    if not args.lang_dir:
        LANG_DIR = os.path.join(RESULTS_DIR, "language") if args.lang is not None else None
    if args.window is not None and args.window != WINDOW:
        # non-default window: separate results dir and auto-LR cache, e.g. lang_sl2_sweep/sl-eng-d04_w20
        WINDOW = args.window
        if not args.results_dir:
            RESULTS_DIR = RESULTS_DIR.rstrip(os.sep) + f"_w{WINDOW}"
    if args.configs: SUBSET = list(args.configs)
    if args.seeds is not None: SEEDS = list(args.seeds)
    if args.epochs: EPOCHS = args.epochs
    if args.n_train: N_TRAIN_SENTENCES = args.n_train
    if args.n_val: N_VAL_SENTENCES = args.n_val
    if args.patience is not None: EARLY_STOP_PATIENCE = args.patience
    if args.no_ckpt: SAVE_CKPT = False
    if args.lr_max or args.lr_points:
        LR_GRID = list(np.geomspace(LR_GRID[0], args.lr_max or LR_GRID[-1], args.lr_points or len(LR_GRID)))
    return args


# main

if __name__ == "__main__":
    args = apply_args(build_parser().parse_args())
    print("=== SL2 Experiment (MULTI_FACT=True, USE_RECALL=True) ===", flush=True)
    print(f"d={HIDDEN_SIZE} window={WINDOW} seq={SEQ_LEN} batch={BATCH_SIZE} "
          f"epochs={EPOCHS}", flush=True)
    print(f"configs={SUBSET}  seeds={SEEDS}", flush=True)
    print(f"lang={LANG_NAME or 'langs/sl2 (default)'} overrides={LANG_OVERRIDES} seed={LANG_SEED}\n"
          f"results -> {RESULTS_DIR}", flush=True)

    sl = load_language()
    tok = WordTokenizer(sl)
    print(f"Vocab size: {len(tok)} (N_GRAMMAR={sl.config.N_GRAMMAR}, "
          f"non_grammar={len(sl.config.non_grammar)}, FACT={sl.config.FACT_DENSITY} "
          f"RECALL={sl.config.RECALL_DENSITY} OTHER={sl.config.OTHER_DENSITY}, "
          f"MULTI_FACT={sl.config.MULTI_FACT}, USE_RECALL={sl.config.USE_RECALL})",
          flush=True)
    save_language_config(sl)
    if args.lang_only:
        print("--lang-only: language ready at", LANG_DIR or os.path.join(_LANGS_DIR, "sl2"), flush=True)
        random.seed(0); np.random.seed(0)
        print("sample:", sl.generate_sentence(60), flush=True)
        sys.exit(0)

    print(f"\nGenerating train corpus ({N_TRAIN_SENTENCES} sentences, "
          f"len={SENTENCE_LEN})...", flush=True)
    train_seqs, _ = generate_corpus(sl, tok, N_TRAIN_SENTENCES, SENTENCE_LEN, seed=999)
    print(f"Train tokens: {sum(len(s) for s in train_seqs):,}", flush=True)

    print(f"Generating val corpus ({N_VAL_SENTENCES} sentences, "
          f"len={SENTENCE_LEN}, with probs)...", flush=True)
    val_seqs, val_probs = generate_corpus(
        sl, tok, N_VAL_SENTENCES, SENTENCE_LEN, seed=888, with_probs=True)
    print(f"Val tokens: {sum(len(s) for s in val_seqs):,}", flush=True)

    recall_pos = find_recall_positions(val_seqs, tok, sl)
    print(f"Recall positions in val: {len(recall_pos)}", flush=True)

    probs_path = os.path.join(RESULTS_DIR, "val_true_probs.npz")
    if not os.path.exists(probs_path):
        os.makedirs(RESULTS_DIR, exist_ok=True)
        np.savez_compressed(probs_path, probs=np.stack(val_probs).astype(np.float16))
        print(f"Saved val true probs to {probs_path}", flush=True)

    print("\nStep 1: auto-LR search", flush=True)
    lrs = {c: FIXED_LR.get(c, FIXED_LR.get("*")) or find_best_lr(c, train_seqs, tok) for c in SUBSET}
    print("\nResolved auto-LR:", flush=True)
    for c in SUBSET:
        print(f"  {c:16s} {RE.CONFIGS[c]}  lr={lrs[c]:.2e}", flush=True)

    plan = [(c, lrs[c], s) for c in SUBSET for s in SEEDS]
    print(f"\nStep 2: {len(plan)} training runs", flush=True)

    for cfg, lr, seed in plan:
        print(f"\n=== {cfg} seed={seed} lr={lr:.2e} ===", flush=True)
        run_single(cfg, lr, seed, tok, sl, train_seqs, val_seqs, val_probs)

    print("\n=== SL2 Experiment done ===", flush=True)
