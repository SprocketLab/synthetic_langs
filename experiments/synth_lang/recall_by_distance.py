#!/usr/bin/env python3
"""Recall accuracy split by distance. For each recall in the per-sentence validation corpus (seed 888,
first SEQ_LEN input tokens), a recall is near if the last occurrence of its fact (definition or earlier
recall) is at most 19 tokens back, within a window-20 attention layer, and far otherwise.
Evaluates the final checkpoint of every configuration and seed in a results directory.

    python recall_by_distance.py --lang-dir sl-eng-d04 --window 20 --results-dir <dir>   # sl-eng
    python recall_by_distance.py --window 20                                                # SL2
    # -> <results dir>/recall_by_distance.json
"""
import os, sys, json, glob
import numpy as np
import torch
import synthetic_lang_sl2 as S

NEAR = 19   # previous tokens visible to a window-20 attention layer


def recall_items(sl, tok, val):
    """(sentence, query position, target token, distance to the most recent occurrence) per recall."""
    n = sl.config.N_FACT_MARK
    fact = {tok.word2id[f"fact{i}"]: i for i in range(1, n + 1) if f"fact{i}" in tok.word2id}
    rec = ({tok.word2id[f"recall{i}"]: i for i in range(1, n + 1) if f"recall{i}" in tok.word2id}
           if sl.config.USE_RECALL else {})
    items = []
    for si, s in enumerate(val):
        input_len = min(len(s) - 1, S.SEQ_LEN)
        defined, last = {}, {}
        for q in range(len(s) - 1):
            t = s[q]
            if sl.config.USE_RECALL:
                if t in fact:
                    defined[fact[t]] = q + 1; last[fact[t]] = q + 1
                elif t in rec and rec[t] in defined:
                    if 3 <= q + 1 <= input_len: items.append((si, q, s[q + 1], q - last[rec[t]]))
                    last[rec[t]] = q + 1
            elif t in fact:
                f = fact[t]
                if f in defined:
                    if 3 <= q + 1 <= input_len: items.append((si, q, s[defined[f]], q - last[f]))
                else:
                    defined[f] = q + 1
                last[f] = q + 1
    return items


def evaluate(model, val, by_sent, attn, device):
    hit = {True: [0, 0], False: [0, 0]}
    with torch.no_grad():
        for i0 in range(0, len(val), 32):
            bs = val[i0:i0 + 32]
            L = min(min(len(s) for s in bs) - 1, S.SEQ_LEN)
            x = torch.stack([torch.tensor(s[:L]) for s in bs]).to(device)
            pred = model(x, attention_mask=attn[:L, :L], return_dict=True)["logits"].argmax(-1).cpu()
            for i in range(len(bs)):
                for q, tg, dist in by_sent.get(i0 + i, []):
                    if q < L:
                        k = dist <= NEAR
                        hit[k][0] += int(pred[i, q].item() == tg); hit[k][1] += 1
    return {"acc_near": hit[True][0] / max(hit[True][1], 1), "acc_far": hit[False][0] / max(hit[False][1], 1),
            "acc_all": (hit[True][0] + hit[False][0]) / max(hit[True][1] + hit[False][1], 1)}


def main():
    args = S.build_parser().parse_args(sys.argv[1:])
    S.apply_args(args)
    device = os.environ.get("DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
    sl = S.load_language(); tok = S.WordTokenizer(sl)
    val, _ = S.generate_corpus(sl, tok, S.N_VAL_SENTENCES, S.SENTENCE_LEN, seed=888)
    items = recall_items(sl, tok, val)
    d = np.array([x[3] for x in items])
    by_sent = {}
    for si, q, tg, dist in items:
        by_sent.setdefault(si, []).append((q, tg, dist))
    attn = torch.ones((S.SEQ_LEN, S.SEQ_LEN))
    attn = (torch.triu(attn, 0) - torch.triu(attn, S.WINDOW)).T.to(device)

    out = {"results_dir": os.path.relpath(S.RESULTS_DIR, S.SWEEP_ROOT), "window": S.WINDOW, "n_recalls": len(d),
           "frac_near": float((d <= NEAR).mean()), "runs": {}}
    for p in sorted(glob.glob(os.path.join(S.RESULTS_DIR, "*", "*seed*.json"))):
        run = json.load(open(p))
        ck = os.path.join(S.RESULTS_DIR, run.get("checkpoint") or "")
        if not os.path.isfile(ck):
            continue
        cfg = run["config"]
        if (args.configs and cfg not in args.configs) or (args.seeds is not None and run["seed"] not in args.seeds):
            continue
        model = S.get_model(S.make_args(cfg, run["lr"]), tok)
        st = torch.load(ck, map_location="cpu", weights_only=True)
        model.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in st.items()}, strict=False)
        model.to(device).eval()
        r = evaluate(model, val, by_sent, attn, device)
        r["seed"] = run["seed"]
        out["runs"].setdefault(cfg, []).append(r)
        print(f"{cfg:16s} seed {run['seed']}  near {r['acc_near']:.3f}  far {r['acc_far']:.3f}  all {r['acc_all']:.3f}",
              flush=True)
        del model; torch.cuda.empty_cache()
    path = os.path.join(S.RESULTS_DIR, os.environ.get("OUT_NAME", "recall_by_distance.json"))
    json.dump(out, open(path, "w"), indent=1)
    print("saved", path)


if __name__ == "__main__":
    main()
