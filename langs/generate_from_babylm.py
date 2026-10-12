"""
Build an sl-eng language from BabyLM: the grammar and non-grammar transition tables come from
trigram counts P(c | a, b) = count(a, b, c) / count(a, b) over contexts seen more than MIN_COUNT
times. Also reports the perplexity exp(H(a, b)) of each context's next-word distribution.

Requires the `datasets` package. Dataset: https://huggingface.co/datasets/BabyLM-community/BabyLM-2026-Strict-Small

    python generate_from_babylm.py                                         -> sl-eng/ (10 grammar words, density 0.01)
    python generate_from_babylm.py --n-grammar 20 --density 0.04 --out sl-eng-ng20
    python generate_from_babylm.py --n-vocab 260 --density 0.04 --out sl-eng-nng250   # 250 non-grammar words
"""

import argparse
import math
import re
import os
import numpy as np
import json
from collections import Counter, defaultdict
from datasets import load_dataset, Dataset, DatasetDict

config = {}

N_FACT_MARK = 20
N_GRAMMAR = 10
N_VOCAB = 1000
FACT_DENSITY = 0.01
RECALL_DENSITY = 0.01

OUT_DIR = "sl-eng"

# ---- Config ----
DATASET_NAME = "BabyLM-community/BabyLM-2026-Strict-Small"
MIN_COUNT = 4  # only keep (a, b) contexts that occur more than this many times

# words of letters/digits; apostrophes only inside a word ("don't")
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)*")


def extract_words(text: str) -> list[str]:
    return [w.lower() for w in WORD_RE.findall(text)]


def main():
    global N_GRAMMAR, N_VOCAB, FACT_DENSITY, RECALL_DENSITY, OUT_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-grammar", type=int, default=N_GRAMMAR)
    ap.add_argument("--n-vocab", type=int, default=N_VOCAB, help="grammar + non-grammar words (AUX included)")
    ap.add_argument("--density", type=float, default=FACT_DENSITY, help="FACT = RECALL density")
    ap.add_argument("--out", default=OUT_DIR, help="output directory, relative to langs/")
    a = ap.parse_args()
    N_GRAMMAR, N_VOCAB, FACT_DENSITY, RECALL_DENSITY = a.n_grammar, a.n_vocab, a.density, a.density
    OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), a.out)
    os.makedirs(OUT_DIR, exist_ok=True)
    # last axis of mc must follow SL2.mc_keys = grammar + ['FACT', 'OTHER', 'RECALL']
    FACT_IDX, OTHER_IDX, RECALL_IDX = N_GRAMMAR, N_GRAMMAR + 1, N_GRAMMAR + 2

    print(f"Loading dataset: {DATASET_NAME}")
    ds = load_dataset(DATASET_NAME)

    mc = np.zeros((N_VOCAB+2, N_VOCAB+2, N_GRAMMAR+3)) + 0.01
    geom = np.zeros((N_VOCAB-N_GRAMMAR, N_VOCAB-N_GRAMMAR))
    geom_counts = np.zeros((N_VOCAB-N_GRAMMAR, N_VOCAB-N_GRAMMAR)) + 1e-4

    token_counts = Counter()
    continuations = defaultdict(Counter)           # (a, b) -> Counter of {c: count}

    # Iterate over every split (train/validation/test, whatever exists)
    for split_name, split_data in ds.items():
        print(f"Processing split: {split_name} ({len(split_data)} rows)")
        # raw text is in the "text" column
        text_column = "text" if "text" in split_data.column_names else split_data.column_names[0]

        for row in split_data:
            tokens = extract_words(row[text_column])

            for t in tokens:
                token_counts[t] += 1

        counts = [(v, i) for v, i in token_counts.items()]
        counts.sort(key=lambda x: -x[1])
        grammar = list([v for v, _ in counts[:N_GRAMMAR]])
        non_grammar = list([v for v, _ in counts[N_GRAMMAR:N_VOCAB-1]]) + ["AUX"]
        vocab = list([v for v, _ in counts[:N_VOCAB-1]]) + ["AUX"]
        t2id = {v[0]: (v[0] if i < N_VOCAB-1 else "AUX") for i, v in enumerate(counts)}
        t2rank = {v[0]: min(i, N_VOCAB-1) for i, v in enumerate(counts)}
        t2rank["FACT"] = -2
        t2rank["RECALL"] = -1

        for row in split_data:
            tokens = extract_words(row[text_column])
            last_non_gram = None

            for a, b, c in zip(tokens, tokens[1:], tokens[2:]):
                if a not in grammar:
                    if True:
                        if t2id[a] == "AUX": continue
                        
                    if last_non_gram:
                        geom_counts[t2rank[last_non_gram]-N_GRAMMAR, t2rank[a]-N_GRAMMAR] += 1
                        geom_counts[t2rank[a]-N_GRAMMAR, t2rank[last_non_gram]-N_GRAMMAR] += 1
                    last_non_gram = a

                r = t2rank[c] if c in grammar else OTHER_IDX

                mc[t2rank[a], t2rank[b], r] += 1
                mc[t2rank["FACT"], t2rank[b], r] += 1
                mc[t2rank["RECALL"], t2rank[b], r] += 1
                mc[t2rank[a], t2rank["FACT"], r] += 1
                mc[t2rank["FACT"], t2rank["FACT"], r] += 1
                mc[t2rank["RECALL"], t2rank["FACT"], r] += 1
                mc[t2rank[a], t2rank["RECALL"], r] += 1
                mc[t2rank["FACT"], t2rank["RECALL"], r] += 1
                mc[t2rank["RECALL"], t2rank["RECALL"], r] += 1

    # grammar/OTHER keep the corpus proportions; FACT/RECALL get a fixed share of
    # every context, as SL2.make_grammar_geom does
    mc[..., FACT_IDX] = 0
    mc[..., RECALL_IDX] = 0
    mc /= np.sum(mc, axis=-1, keepdims=True)
    mc *= 1 - FACT_DENSITY - RECALL_DENSITY
    mc[..., FACT_IDX] = FACT_DENSITY
    mc[..., RECALL_IDX] = RECALL_DENSITY

    geom = geom_counts / np.sum(geom_counts, axis=-1, keepdims=True)
    geom = -np.log(geom)

    np.save(os.path.join(OUT_DIR, "mc"), mc)
    np.save(os.path.join(OUT_DIR, "geom"), geom)



    config["N_FACT_MARK"] = N_FACT_MARK
    config["N_VOCAB"] = N_VOCAB
    config["N_GRAMMAR"] = N_GRAMMAR
    config["FACT_DENSITY"] = FACT_DENSITY
    config["RECALL_DENSITY"] = RECALL_DENSITY
    config["GRAMMAR_ORDER"] = 2
    config["MC_CLOSE_TEMP"] = -1
    config["MC_FAR_TEMP"] = -1
    config["DOUBLE_GRAMMAR_FACT"] = 1.0
    config["RW_TEMP"] = 1.0
    # config["USE_PRIOR"] = True
    config["USE_PRIOR"] = False
    config["MULTI_FACT"] = False
    config["USE_RECALL"] = False
    config["NON_GRAM_TYPE"] = "rw"
    config["GRAMMAR_SIGMA"] = -1
    config["NONGRAMMAR_SIGMA"] = -1

    config["vocab"] = vocab
    config["grammar"] = grammar
    config["non_grammar"] = non_grammar
    config["all_vocab"] = vocab + ["<start>"] + ['fact%d' % i for i in range(1, N_FACT_MARK+1)]
    # config["reserved"] = 

    prior = np.ones(len(config["all_vocab"]))
    for i, t in enumerate(config["vocab"]):
        prior[i] = token_counts[t]
    prior[:len(config["vocab"])] /= np.sum(prior[:len(config["vocab"])])

    config["OTHER_DENSITY"] = np.sum(prior[N_GRAMMAR:N_VOCAB])
    config["prior"] = prior.tolist()

    with open(os.path.join(OUT_DIR, "config.json"), "w") as outfile:
        json.dump(config, outfile, indent=4)



if __name__ == "__main__":
    main()
