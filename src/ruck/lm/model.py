"""A small causal transformer over phase-play tokens.

Deliberately small. The corpus a club can assemble is tens of thousands of
events, not billions, so the budget in the spec (10M parameters) is an upper
bound rather than a target: the default configuration is about 1M parameters,
which is already more capacity than a season of tagged matches supports.
:meth:`PhaseLM.n_params` is checked against the cap at construction so a
configuration cannot quietly drift past it.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ruck.tokenizer import PAD

MAX_PARAMS = 10_000_000


@dataclass
class LMConfig:
    """Model shape. Defaults are sized for a season-scale corpus."""

    vocab_size: int
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    d_ff: int = 512
    max_len: int = 256
    dropout: float = 0.1
    tie_weights: bool = True

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")


class CausalSelfAttention(nn.Module):
    def __init__(self, config: LMConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model)
        self.proj = nn.Linear(config.d_model, config.d_model)
        self.dropout = config.dropout

    def forward(self, x: Tensor) -> Tensor:
        b, t, c = x.shape
        q, k, v = self.qkv(x).split(c, dim=2)
        shape = (b, t, self.n_heads, self.head_dim)
        q, k, v = (z.view(shape).transpose(1, 2) for z in (q, k, v))
        # Padding sits to the right of every sequence and attention is causal,
        # so a real query never reaches a pad key and no key mask is needed.
        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=True
        )
        return self.proj(out.transpose(1, 2).contiguous().view(b, t, c))


class Block(nn.Module):
    """Pre-norm transformer block."""

    def __init__(self, config: LMConfig):
        super().__init__()
        self.norm_attention = nn.LayerNorm(config.d_model)
        self.attention = CausalSelfAttention(config)
        self.norm_ff = nn.LayerNorm(config.d_model)
        self.ff = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Linear(config.d_ff, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attention(self.norm_attention(x))
        return x + self.ff(self.norm_ff(x))


class PhaseLM(nn.Module):
    """Next-event language model over ``ruck.tokenizer`` tokens."""

    def __init__(self, config: LMConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model, padding_idx=PAD)
        self.position_embedding = nn.Embedding(config.max_len, config.d_model)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.n_layers))
        self.norm = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_weights:
            self.head.weight = self.token_embedding.weight
        self.apply(self._init)

        total = self.n_params()
        if total > MAX_PARAMS:
            raise ValueError(f"{total:,} parameters exceeds the {MAX_PARAMS:,} budget")

    @staticmethod
    def _init(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def n_params(self, trainable_only: bool = False) -> int:
        """Parameter count, counting a tied embedding once."""
        seen: set[int] = set()
        total = 0
        for parameter in self.parameters():
            if id(parameter) in seen:
                continue
            if trainable_only and not parameter.requires_grad:
                continue
            seen.add(id(parameter))
            total += parameter.numel()
        return total

    def forward(self, ids: Tensor) -> tuple[Tensor, Tensor]:
        """Return ``(logits, hidden)`` for a right-padded batch of ids."""
        _, t = ids.shape
        if t > self.config.max_len:
            raise ValueError(f"sequence of {t} exceeds max_len {self.config.max_len}")
        positions = torch.arange(t, device=ids.device)
        x = self.drop(self.token_embedding(ids) + self.position_embedding(positions))
        for block in self.blocks:
            x = block(x)
        hidden = self.norm(x)
        return self.head(hidden), hidden

    @torch.no_grad()
    def token_surprisal(self, ids: Tensor) -> Tensor:
        """Surprisal in bits for each position, given everything before it.

        Position 0 has no context, so its surprisal is ``nan`` rather than a
        number that would quietly become part of an average.
        """
        self.eval()
        logits, _ = self(ids)
        log_probs = F.log_softmax(logits[:, :-1].float(), dim=-1)
        targets = ids[:, 1:]
        nats = -log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        bits = nats / math.log(2.0)
        pad = torch.full((ids.shape[0], 1), float("nan"), device=ids.device)
        return torch.cat([pad, bits], dim=1)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"config": asdict(self.config), "state": self.state_dict()}, path)

    @classmethod
    def load(cls, path: str | Path, map_location: str = "cpu") -> PhaseLM:
        blob = torch.load(path, map_location=map_location, weights_only=False)
        model = cls(LMConfig(**blob["config"]))
        model.load_state_dict(blob["state"])
        model.eval()
        return model
