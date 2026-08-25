"""Training loop for the phase-play language model.

Small enough to be honest about: AdamW, a linear warmup into cosine decay,
early stopping on validation perplexity, and a fixed seed. There is no
learning-rate search and no ensembling; the point of this model is to produce
calibrated surprisal for a report, not to win anything.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from ruck.lm.data import count_truncated, iter_batches, pick_device
from ruck.lm.metrics import PerplexityResult, model_perplexity
from ruck.lm.model import LMConfig, PhaseLM
from ruck.tokenizer import Encoded


@dataclass
class TrainConfig:
    epochs: int = 25
    batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    grad_clip: float = 1.0
    patience: int = 5
    seed: int = 0
    device: str = "auto"
    log_every: int = 0  # 0 = once per epoch


@dataclass
class TrainHistory:
    """What happened during training, for the README and the report footer."""

    train_loss: list[float] = field(default_factory=list)
    val_perplexity: list[float] = field(default_factory=list)
    best_epoch: int = -1
    best_val_perplexity: float = float("inf")
    seconds: float = 0.0
    n_params: int = 0
    truncated_sequences: int = 0

    def summary(self) -> str:
        return (
            f"{self.n_params:,} params, best val ppl {self.best_val_perplexity:.3f} "
            f"at epoch {self.best_epoch + 1}/{len(self.val_perplexity)} "
            f"in {self.seconds:.1f}s"
        )


def train(
    sequences_train: list[Encoded],
    sequences_val: list[Encoded],
    *,
    vocab_size: int,
    model_config: LMConfig | None = None,
    config: TrainConfig | None = None,
    checkpoint: str | Path | None = None,
    verbose: bool = True,
) -> tuple[PhaseLM, TrainHistory]:
    """Train a :class:`PhaseLM` and return the best-validation checkpoint."""
    config = config or TrainConfig()
    model_config = model_config or LMConfig(vocab_size=vocab_size)
    if model_config.vocab_size != vocab_size:
        raise ValueError("model_config.vocab_size disagrees with the tokenizer")

    torch.manual_seed(config.seed)
    device = pick_device(config.device)
    model = PhaseLM(model_config).to(device)

    history = TrainHistory(
        n_params=model.n_params(),
        truncated_sequences=count_truncated(sequences_train, model_config.max_len),
    )
    if verbose and history.truncated_sequences:
        print(
            f"  {history.truncated_sequences} training sequence(s) longer than "
            f"max_len={model_config.max_len} were truncated"
        )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    steps_per_epoch = max(1, math.ceil(len(sequences_train) / config.batch_size))
    total_steps = steps_per_epoch * config.epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: _lr_scale(step, config.warmup_steps, total_steps)
    )

    best_state: dict[str, torch.Tensor] | None = None
    started = time.monotonic()
    since_improvement = 0

    for epoch in range(config.epochs):
        model.train()
        running = 0.0
        seen = 0
        for ids, mask in iter_batches(
            sequences_train,
            batch_size=config.batch_size,
            max_len=model_config.max_len,
            shuffle=True,
            seed=config.seed + epoch,
        ):
            ids = ids.to(device)
            mask = mask.to(device)
            logits, _ = model(ids)
            targets = ids[:, 1:]
            valid = mask[:, 1:]
            losses = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.shape[-1]),
                targets.reshape(-1),
                reduction="none",
            ).view(targets.shape)
            n_valid = valid.sum().clamp(min=1)
            loss = (losses * valid).sum() / n_valid

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            scheduler.step()

            running += float(loss.item()) * int(n_valid.item())
            seen += int(n_valid.item())

        train_loss = running / max(seen, 1)
        val = model_perplexity(
            model, sequences_val, batch_size=config.batch_size, device=device, name="val"
        )
        history.train_loss.append(train_loss)
        history.val_perplexity.append(val.perplexity)

        if val.perplexity < history.best_val_perplexity - 1e-4:
            history.best_val_perplexity = val.perplexity
            history.best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            since_improvement = 0
        else:
            since_improvement += 1

        if verbose:
            print(
                f"  epoch {epoch + 1:>3}/{config.epochs}  "
                f"train loss {train_loss:.4f}  val ppl {val.perplexity:.3f}"
                + ("  *" if history.best_epoch == epoch else "")
            )
        if since_improvement >= config.patience:
            if verbose:
                print(f"  early stop: no improvement in {config.patience} epochs")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to("cpu").eval()
    history.seconds = time.monotonic() - started

    if checkpoint is not None:
        model.save(checkpoint)
    return model, history


def evaluate(
    model: PhaseLM,
    sequences: list[Encoded],
    *,
    name: str = "test",
    batch_size: int = 64,
    device: str = "auto",
) -> PerplexityResult:
    return model_perplexity(
        model, sequences, batch_size=batch_size, device=pick_device(device), name=name
    )


def _lr_scale(step: int, warmup: int, total: int) -> float:
    if warmup > 0 and step < warmup:
        return (step + 1) / warmup
    if total <= warmup:
        return 1.0
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
