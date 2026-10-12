# Lightweight Experimentation with Nested Markov Chains

Official code for the paper *Lightweight Experimentation with Nested Markov Chains*.

[Paper (coming soon)]() | [Citation](#citation)

We study synthetic languages as a cheap testbed for hybrid sequence models. Each language is a
nested Markov chain: a second-order grammar chain, an embedded first-order chain over
non-grammar words, and fact-recall pairs that create long-range dependencies. **SL2** is generated
at random with a small vocabulary; **sl-eng** takes its statistics from BabyLM. We train 8-layer
hybrids built from Mamba (`S`), sliding-window attention (`T`) and DeltaProduct (`D`) layers on these
languages and compare their rankings with training on English.

This repository contains:

- the SL2 and sl-eng language generators, plus the exact languages used in the paper;
- the hybrid model implementation and training code;
- scripts to reproduce every experiment, table and figure in the paper.

## Installation

The code was tested with Python 3.11, PyTorch 2.8.0 and CUDA 12.8.

```bash
git clone https://github.com/SprocketLab/synthetic_langs.git
cd synthetic_langs

conda create -n synthlang python=3.11 -y
conda activate synthlang

pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

# prebuilt Mamba kernels for torch 2.8 / Python 3.11
pip install --no-deps \
  https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.6.2.post1/causal_conv1d-1.6.2.post1+cu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl \
  https://github.com/state-spaces/mamba/releases/download/v2.3.2.post1/mamba_ssm-2.3.2.post1+cu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl

export WANDB_MODE=disabled
```

`transformers` must be below 5.0. Our `S` layers run Mamba without the short causal convolution
(`do_conv = False` in `experiments/models/mamba.py`).

## Data

The languages used in the paper are included:

| Language | Location | Vocabulary |
|---|---|---|
| SL2 | `langs/sl2/` | 116 |
| sl-eng | `langs/sl-eng-d04.tar.gz` | 1021 |

Unpack sl-eng once before running any sl-eng experiment:

```bash
cd langs && tar -xzf sl-eng-d04.tar.gz
```

BabyLM is downloaded from the Hugging Face Hub on first use.

## Quick start

Train two architectures on SL2 for one epoch with a fixed learning rate:

```bash
cd experiments/synth_lang
python synthetic_lang_sl2.py --window 20 --configs pure_tf 6s2t_start --seeds 0 --epochs 1 --lr 1e-3
```

Each run writes one JSON file per configuration and seed to `results/synthetic/lang_sl2_w20/`, with
validation perplexity, KL divergence to the true distribution by word type, and recall accuracy for
every epoch.

## Architectures

Layer layouts are defined in `experiments/decode_recall.py` (first layer on the left).

| Name | Layers | Name | Layers |
|---|---|---|---|
| `pure_tf` | TTTTTTTT | `7s1t_mid` | SSSTSSSS |
| `4s4t_alternate` | STSTSTST | `pure_ssm` | SSSSSSSS |
| `4s4t_tf_first` | TTTTSSSS | `pure_dp` | DDDDDDDD |
| `4s4t_ssm_first` | SSSSTTTT | `4d4t_alternate` | DTDTDTDT |
| `6s2t_start` | TTSSSSSS | `6d2t_start` | TTDDDDDD |
| `6s2t_end` | SSSSSSTT | `dsts_alternate` | DSTSDSTS |
| `6s2t_even` | SSSTSSST | | |

The commands below use all 13:

```bash
CFGS13="pure_tf 4s4t_alternate 4s4t_tf_first 4s4t_ssm_first 6s2t_start 6s2t_end 6s2t_even 7s1t_mid pure_ssm pure_dp 4d4t_alternate 6d2t_start dsts_alternate"
```

## Reproducing the paper

Run each script from its own directory. Results go to `results/synthetic/`, and tables and figures
to `figures/`.

| Paper | Script |
|---|---|
| Sec. 3, associative recall vs. BabyLM | `synthetic_sweeps/mkar_d256_full.py`, `synthetic_sweeps/decode_recall_d256_full.py`, `posttrain/pretrain_babylm.py` |
| Sec. 5.1, Table 1 | `synth_lang/synthetic_lang_sl2.py`, `synth_lang/sleng_fill_table1.py`, `synth_lang/make_paper_tables.py` |
| Sec. 5.2, longer training on sl-eng | `synth_lang/synthetic_lang_sl2.py`, `synth_lang/plot_round2_figs.py` |
| Sec. 5.3, language parameters | `synth_lang/sl2_sweep.py`, `synth_lang/synthetic_lang_sl2.py`, `synth_lang/plot_paper_figs.py` |
| Sec. 6, transfer and pretraining | `posttrain/pretrain_babylm.py`, `posttrain/posttrain_sl2_v2.py`, `posttrain/posttrain_stage3.py` |
| Appendix, full attention | `synth_lang/synthetic_lang_sl2.py`, `synth_lang/plot_paper_figs.py` |

### Section 3: associative recall vs. BabyLM

```bash
cd experiments/synthetic_sweeps
python mkar_d256_full.py            # MKAR and MQAR, d=256, 5 seeds
python decode_recall_d256_full.py   # decode-recall, d=256, 5 seeds
cd ../posttrain
python pretrain_babylm.py --sources none --configs $CFGS13   # BabyLM from scratch, 3 seeds
```

### Section 5.1: Table 1

SL2 trains on 20k sentences for 15 epochs. sl-eng trains on 100k sentences for 3 epochs, the same
number of tokens.

```bash
cd experiments/synth_lang
python synthetic_lang_sl2.py --window 20 --configs $CFGS13
for c in $CFGS13; do python sleng_fill_table1.py $c; done
python recall_by_distance.py --window 20
python make_paper_tables.py components components_20k
```

`sleng_fill_table1.py` first runs 20k sentences x 15 epochs with the learning-rate search (the
appendix table), then 100k sentences x 3 epochs at the learning rate it found (Table 1).

### Section 5.2: longer training on sl-eng

```bash
cd experiments/synth_lang
# LR: the value for config $c in results/synthetic/lang_sl2_sweep/sl-eng-d04_w20/auto_lr.json
python synthetic_lang_sl2.py --lang-dir sl-eng-d04 --window 20 --patience 0 --n-train 100000 \
  --results-dir ../../results/synthetic/lang_sl2_sweep/sl-eng-d04_w20_n100k --configs $c --lr $c=$LR
python plot_round2_figs.py long_training
```

### Section 5.3: language parameters

```bash
cd experiments/synth_lang
python sl2_sweep.py list                               # grid points: ng*, nng*, fr*
python synthetic_lang_sl2.py --lang ng5 --window 20    # one run per grid point
python plot_paper_figs.py sensitivity
python plot_round2_figs.py sensitivity_kl
```

Each grid language is generated from a fixed seed (`--lang-seed`, default 0) and saved next to its
results.

### Section 6: transfer to BabyLM, pretraining and pre-pretraining

```bash
cd experiments/synth_lang
python plot_rank_slope.py --pair sl2 babylm        # SL2 vs. BabyLM rankings

cd ../posttrain
python pretrain_babylm.py --sources sl2            # SL2 -> BabyLM
python posttrain_sl2_v2.py --tasks mkar mqar --variants full scratch_matched   # SL2 -> task
python posttrain_stage3.py --sources sl2 none      # (SL2 ->) BabyLM -> MKAR / MQAR
cd ../synth_lang
python plot_round2_figs.py prepretrain epoch20
python plot_paper_figs.py pipelines
```

For pretraining, each initialisation is also trained at the other's learning rate
(`pretrain_babylm.py --lr CONFIG=LR --results-dir ../../results/synthetic/prepretrain_lrswap`).
`configs/stage2_manifest.json` lists, for each configuration, the BabyLM checkpoint with the lower
perplexity, which the last stage starts from.

### Appendix: full attention

```bash
cd experiments/synth_lang
python synthetic_lang_sl2.py --window 200 --no-ckpt
python synthetic_lang_sl2.py --lang-dir sl-eng-d04 --window 200 --patience 0 --lr-max 3e-3 --lr-points 10 --no-ckpt
python plot_paper_figs.py window_ablation
```

## Generating new languages

```bash
cd langs
python generate_from_babylm.py --density 0.04 --out sl-eng-new                  # sl-eng from BabyLM
cd ../experiments/synth_lang
python synthetic_lang_sl2.py --lang my --set N_GRAMMAR=15 --set FACT_DENSITY=0.02 --lang-only   # SL2 variant
```

`generate_from_babylm.py` samples part of the language at random, so a regenerated sl-eng has the
same configuration as ours but different transition tables. Use `sl-eng-d04.tar.gz` to reproduce
the paper.

## Repository structure

```
langs/
  sl2.py                    SL2 generator
  generate_from_babylm.py   sl-eng generator
  sl2/, sl-eng-d04.tar.gz   languages used in the paper
experiments/
  models/                   hybrid stack, Mamba, attention and DeltaProduct layers
  decode_recall.py          layer configurations, auto-LR search, sweep runner
  generate.py, data_utils.py, train_utils.py, test_utils.py, model_utils.py, babylm_data.py
  synthetic_sweeps/         MKAR, MQAR and decode-recall sweeps
  synth_lang/               training on SL2 / sl-eng, analysis, tables and figures
  posttrain/                BabyLM training and the pretraining pipelines
configs/
  stage2_manifest.json      BabyLM checkpoints used for the last pipeline stage
```

## Citation

```bibtex
@misc{synthetic_langs,
  title  = {Lightweight Experimentation with Nested Markov Chains},
  note   = {Citation will be added after publication},
}
```
