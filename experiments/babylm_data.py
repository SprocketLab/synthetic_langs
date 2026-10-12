#!/usr/bin/env python3
"""BabyLM loading and batching."""
import numpy as np
import torch


def load_babylm_tokens(tokenizer, split="train", max_tokens=None):
    """Load babyLM strict-small (10M words), tokenize, return flat int64 array."""
    from datasets import load_dataset
    ds = load_dataset("nilq/babylm-10M", split=split)
    eos = tokenizer.eos_token_id

    all_ids = []
    for row in ds:
        text = row["text"].strip()
        if not text:
            continue
        all_ids.extend(tokenizer.encode(text))
        all_ids.append(eos)
        if max_tokens and len(all_ids) >= max_tokens:
            all_ids = all_ids[:max_tokens]
            break

    print(f"  [{split}] {len(all_ids):,} tokens from {len(ds):,} rows")
    return np.array(all_ids, dtype=np.int64)


class TextDataset:
    """Fixed-length chunks of tokenized text, batched internally.

    __getitem__(i) returns a dict of input_ids, output_ids (shifted by one) and mask,
    each [batch_size, seq_len], as train_epoch expects.
    """

    def __init__(self, token_ids, seq_len, batch_size, shuffle=True):
        n_chunks = len(token_ids) // (seq_len + 1)
        token_ids = token_ids[:n_chunks * (seq_len + 1)]
        self.chunks = token_ids.reshape(n_chunks, seq_len + 1)
        if shuffle:
            np.random.shuffle(self.chunks)
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.n_batches = n_chunks // batch_size
        self.chunks = self.chunks[:self.n_batches * batch_size]

    def __len__(self):
        return self.n_batches

    def __getitem__(self, idx):
        if idx >= self.n_batches:
            raise IndexError(idx)
        start = idx * self.batch_size
        batch = self.chunks[start:start + self.batch_size]
        input_ids = torch.tensor(batch[:, :-1], dtype=torch.long)
        output_ids = torch.tensor(batch[:, 1:], dtype=torch.long)
        mask = torch.ones_like(output_ids)
        return {"input_ids": input_ids, "output_ids": output_ids, "mask": mask}
