"""Minimal LoRA for the hybrid backbone.

LoRALinear exposes `.weight` / `.bias` as properties returning the merged weight
W + (alpha/r) * B @ A, because the Mamba fast path reads x_proj / dt_proj / out_proj
weights directly instead of calling the module.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 8, alpha: float = 16.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.r = r
        self.scale = alpha / r
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features,
                                               device=base.weight.device,
                                               dtype=base.weight.dtype))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r,
                                               device=base.weight.device,
                                               dtype=base.weight.dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    @property
    def in_features(self):
        return self.base.in_features

    @property
    def out_features(self):
        return self.base.out_features

    @property
    def delta(self):
        return self.scale * (self.lora_B @ self.lora_A)

    @property
    def weight(self):
        return self.base.weight + self.delta

    @property
    def bias(self):
        return self.base.bias

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)


def apply_lora(model, r=8, alpha=16.0, train_embeddings=True):
    """Wrap every nn.Linear in model.hybrid.layers with LoRALinear and freeze the backbone.

    Trainable: the LoRA factors and, with train_embeddings, embed_in / embed_out and the final
    layer norm. Returns the number of wrapped layers.
    """
    for p in model.parameters():
        p.requires_grad_(False)

    n_wrapped = 0
    for layer in model.hybrid.layers:
        for name, mod in list(layer.named_modules()):
            for child_name, child in list(mod.named_children()):
                if isinstance(child, nn.Linear):
                    setattr(mod, child_name, LoRALinear(child, r=r, alpha=alpha))
                    n_wrapped += 1

    if train_embeddings:
        for p in model.hybrid.embed_in.parameters():
            p.requires_grad_(True)
        for p in model.embed_out.parameters():
            p.requires_grad_(True)
        for p in model.hybrid.final_layer_norm.parameters():
            p.requires_grad_(True)
    return n_wrapped


def effective_state_dict(model, include_buffers=False):
    """Parameters (LoRA merged, original names) as a detached dict.

    Buffers are excluded unless include_buffers=True (needed for a loadable state dict).
    Also works for models without LoRA.
    """
    param_names = set(n for n, _ in model.named_parameters(remove_duplicate=False))
    lora_mods = {n: m for n, m in model.named_modules() if isinstance(m, LoRALinear)}
    sd = {}
    for k, v in model.state_dict().items():
        if ".lora_A" in k or ".lora_B" in k:
            continue
        if not include_buffers and k not in param_names:
            continue
        k2 = k.replace(".base.weight", ".weight").replace(".base.bias", ".bias")
        sd[k2] = v.detach().clone()
    with torch.no_grad():
        for n, m in lora_mods.items():
            sd[n + ".weight"] = m.weight.detach().clone()
    return sd


def trainable_parameters(model):
    return [p for p in model.parameters() if p.requires_grad]


def count_params(model, trainable_only=False):
    return sum(p.numel() for p in model.parameters()
               if (p.requires_grad or not trainable_only))
