"""Every knob for a training run, in one place."""

from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class TrainConfig:
    # --- model ---
    vocab_size: int = 50304      # padded past the tokenizer's 50257 for kernel alignment
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    block_size: int = 1024
    dropout: float = 0.0         # one pass over unique tokens: nothing to regularize
    bias: bool = True

    # --- batch ---
    B: int = 2                   # micro-batch; measured optimal on 16GB MPS
    T: int = 1024
    total_batch_size: int = 16384    # tokens per optimizer step, before grad accum

    # --- data window: shards[first_shard : first_shard + num_shards] ---
    first_shard: int = 3
    num_shards: int = 4
    data_root: Optional[str] = None
    num_epoch: int = 1
    max_steps: Optional[int] = None   # None = derive from the data window; set to cap a smoke run

    # --- optimizer / schedule ---
    max_lr: float = 6e-4
    min_lr_ratio: float = 0.1
    warmup_steps: int = 100
    # How many steps the cosine spans. None = this run's own end, which makes each
    # continuation a warm restart: the LR climbs back toward max_lr and decays again.
    # For one uninterrupted decay across several windows, set this to the total steps
    # you plan to train for and leave it fixed across every run.
    lr_schedule_steps: Optional[int] = None
    weight_decay: float = 0.1
    betas: Tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 1.0

    # --- eval / logging ---
    eval_every: int = 300
    eval_batches: int = 20
    sample_prompt: str = 'hello, I am a language model, '
    sample_sequences: int = 5
    sample_length: int = 50
    tokenizer_vocab: int = 50257     # real decodable vocab; logits are clipped to this

    # --- run artifacts ---
    run_dir: Optional[str] = None
    resume_from: Optional[str] = 'latest'
    keep_last_n_checkpoints: Optional[int] = 3
    # Restore the model+optimizer but NOT the data position -- start at the front of
    # this window. Set True when moving to a new shard window with a checkpoint saved
    # before shard names were recorded, since those cannot be checked against the
    # window and would otherwise resume at an arbitrary offset into the new data.
    reset_data_position: bool = False

    # --- backend switches ---
    # Both are wins on CUDA and losses on MPS, measured at B=2 T=1024 batch=16384:
    # compile+bf16 985 tok/s, eager+bf16 1560, eager+fp32 1820. MPS has no native
    # bf16 path and the inductor MPS backend loses to eager. None = auto (CUDA only).
    use_compile: Optional[bool] = None
    use_amp: Optional[bool] = None
    seed: int = 42

    @property
    def min_lr(self) -> float:
        return self.max_lr * self.min_lr_ratio

    def model_kwargs(self) -> dict:
        return dict(vocab_size=self.vocab_size, n_layer=self.n_layer, n_head=self.n_head,
                    n_embd=self.n_embd, block_size=self.block_size,
                    dropout=self.dropout, bias=self.bias)
