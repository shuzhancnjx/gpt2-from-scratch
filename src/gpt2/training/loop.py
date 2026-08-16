"""Assembles and runs a training loop.

train() takes its model and loaders as arguments. That is deliberate: fine-tuning
differs from pretraining in the dataset, the loss mask, and the schedule -- not in
the loop -- so gpt2.finetune can reuse this instead of forking it.
"""

import glob
import math
import os
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from gpt2.data import DataLoaderLite
from gpt2.gpt import GPT, GPTConfig
from gpt2.log import DEFAULT_RUN_DIR, Logger, load_checkpoint
from gpt2.training.config import TrainConfig
from gpt2.training.runtime import Runtime, setup_runtime


def get_lr(step, max_steps, max_lr, min_lr, warmup_steps):
    """Linear warmup, then cosine decay to min_lr.

    max_steps is the span the cosine is stretched over, which on a continuation is
    the cumulative total across every run so far -- not just this run's length. See
    TrainConfig.lr_schedule_steps.
    """
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step > max_steps:
        return min_lr

    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


def resolve_checkpoint(spec, run_dir):
    """Turn a resume_from value into a real path.

    Accepts, in order of preference:
      'latest'                     -> newest model_*.pt in run_dir
      '/abs/path/model_001830.pt'  -> used as given
      'model_001830.pt'            -> looked up inside run_dir
      'runs/model_001830.pt'       -> tried as given (cwd-relative), then in run_dir

    The run_dir fallback exists so a config does not break just because the run was
    launched from a different directory.
    """
    if spec == 'latest':
        # tolerant by design: 'latest' means "continue if there is something to
        # continue from", so a first run in a fresh run_dir starts from scratch.
        # An explicit path that is missing still raises -- a typo must not silently
        # restart training.
        found = sorted(glob.glob(os.path.join(run_dir, 'model_*.pt')))
        return found[-1] if found else None

    if os.path.exists(spec):
        return os.path.abspath(spec)

    candidate = os.path.join(run_dir, os.path.basename(spec))
    if os.path.exists(candidate):
        return candidate

    raise FileNotFoundError(f'no checkpoint at {spec!r} (also tried {candidate})')


@torch.no_grad()
def evaluate(model, loader, batches, runtime):
    """Mean loss over `batches` batches, averaged across ranks. Returns a float."""
    was_training = model.training
    model.eval()

    total = torch.zeros((), device=runtime.device)
    for _ in range(batches):
        x, y = loader.next_batch()
        with runtime.amp_ctx():
            _, loss = model(x, y)
        total += (loss / batches).detach()

    if runtime.ddp:
        dist.all_reduce(total, op=dist.ReduceOp.AVG)

    if was_training:
        model.train()
    return total.item()


def build_loaders(cfg: TrainConfig, runtime: Runtime):
    train_loader = DataLoaderLite(
        B=cfg.B, T=cfg.T, device=runtime.device, process_rank=runtime.rank,
        num_processes=runtime.world_size, split='train',
        master_process=runtime.master_process,
        start_shard=cfg.first_shard, max_shards=cfg.num_shards, data_root=cfg.data_root)
    val_loader = DataLoaderLite(
        B=cfg.B, T=cfg.T, device=runtime.device, process_rank=runtime.rank,
        num_processes=runtime.world_size, split='val',
        master_process=runtime.master_process, data_root=cfg.data_root)
    return train_loader, val_loader


def build_model(cfg: TrainConfig, runtime: Runtime):
    """Returns (model, raw_model). raw_model is the unwrapped module -- use it for
    the optimizer, sampling, and checkpoints; DDP does not forward attributes and
    a compiled module prefixes its state_dict keys."""
    raw_model = GPT(GPTConfig(**cfg.model_kwargs())).to(runtime.device)
    model = raw_model
    if runtime.use_compile:
        model = torch.compile(model)
    if runtime.ddp:
        model = DDP(model, device_ids=[runtime.local_rank])
    return model, raw_model


def train(cfg: TrainConfig, runtime: Runtime = None, model=None, raw_model=None,
          train_loader=None, val_loader=None, logger=None):
    runtime = runtime or setup_runtime(cfg.use_compile, cfg.use_amp, cfg.seed)

    if train_loader is None or val_loader is None:
        train_loader, val_loader = build_loaders(cfg, runtime)
    if model is None:
        model, raw_model = build_model(cfg, runtime)
    raw_model = raw_model or model

    assert cfg.total_batch_size % (cfg.B * cfg.T * runtime.world_size) == 0, (
        f'total_batch_size {cfg.total_batch_size} not divisible by '
        f'B*T*world_size {cfg.B * cfg.T * runtime.world_size}')
    grad_accum_steps = cfg.total_batch_size // (cfg.B * cfg.T * runtime.world_size)

    optimizer = raw_model.configure_optimizers(
        weight_decay=cfg.weight_decay, learning_rate=cfg.max_lr,
        betas=cfg.betas, device_type=runtime.device_type)

    # resolve before building the Logger: whether we are *actually* resuming decides
    # whether the log is appended to or truncated
    run_dir = cfg.run_dir or DEFAULT_RUN_DIR
    resume_path = resolve_checkpoint(cfg.resume_from, run_dir) if cfg.resume_from else None
    if cfg.resume_from == 'latest' and resume_path is None:
        runtime.print0(f'=> resume_from="latest": no checkpoint in {run_dir}, starting fresh')

    logger = logger or Logger(run_dir=run_dir, master_process=runtime.master_process,
                              resuming=resume_path is not None,
                              keep_last_n=cfg.keep_last_n_checkpoints)

    start_step = 0
    if resume_path is not None:
        ckpt = load_checkpoint(resume_path, map_location=runtime.device)
        raw_model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        if cfg.reset_data_position:
            runtime.print0('=> reset_data_position: ignoring the checkpoint data offset, '
                           f'starting at {os.path.basename(train_loader.shards[0])}')
        else:
            train_loader.load_state_dict(ckpt['train_loader'])
        start_step = ckpt['step'] + 1
        runtime.print0(f'=> resumed from {resume_path} at step {start_step}')

    # Steps this window is worth. A continuation trains its whole window, so the
    # loop ends at start_step + that -- deriving max_steps from the window alone
    # would silently cut the run short by however far the checkpoint had already got.
    steps_this_run = cfg.num_epoch * train_loader.total_num_tokens // cfg.total_batch_size
    max_steps = cfg.max_steps or (start_step + steps_this_run)
    # The cosine spans lr_schedule_steps; leaving it None makes each continuation a
    # warm restart (LR climbs back up, then decays). Set it to the total steps across
    # every planned window for one uninterrupted decay instead.
    lr_span = cfg.lr_schedule_steps or max_steps

    runtime.print0(f'=> total desired batch size: {cfg.total_batch_size}')
    runtime.print0(f'=> calculated gradient accumulated steps: {grad_accum_steps}')
    runtime.print0(f'=> training steps: {steps_this_run} '
                   f'(step {start_step} -> {max_steps}), lr schedule spans {lr_span}')
    runtime.print0(f'=> lr at start: {get_lr(start_step, lr_span, cfg.max_lr, cfg.min_lr, cfg.warmup_steps):.2e}'
                   f'  -> end: {cfg.min_lr:.2e}')

    enc = None
    if cfg.sample_prompt and runtime.master_process:
        import tiktoken
        enc = tiktoken.get_encoding('gpt2')

    for step in range(start_step, max_steps):
        last_step = (step == max_steps - 1)

        # --- eval runs alongside the training step, never instead of it ---
        if (step > 0 and step % cfg.eval_every == 0) or last_step:
            val_loss = evaluate(model, val_loader, cfg.eval_batches, runtime)
            runtime.print0(f'======> validation loss: {val_loss:.4f}')
            logger.log(step, 'val', val_loss)

            if runtime.master_process:
                if enc is not None:
                    _sample(raw_model, enc, cfg, runtime)
                logger.save_checkpoint(step, raw_model, optimizer, train_loader)

        # --- train ---
        t0 = time.time()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_accum = torch.zeros((), device=runtime.device)

        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch()
            with runtime.amp_ctx():
                _, loss = model(x, y)

            loss = loss / grad_accum_steps
            loss_accum += loss.detach()

            if runtime.ddp and micro_step < grad_accum_steps - 1:
                with model.no_sync():
                    loss.backward()
            else:
                loss.backward()

        if runtime.ddp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

        lr = get_lr(step, lr_span, cfg.max_lr, cfg.min_lr, cfg.warmup_steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        optimizer.step()
        runtime.synchronize()

        dt = (time.time() - t0)
        tokens_per_sec = (cfg.B * cfg.T * grad_accum_steps * runtime.world_size) / dt
        runtime.print0(f'step {step}, loss: {loss_accum.item():.4f}, norm: {norm:.4f}, '
                       f'lr: {lr:.6f}, dt: {dt * 1000:.2f}ms, token/sec: {tokens_per_sec:.2f}')
        logger.log(step, 'train', loss_accum.item())

    runtime.shutdown()
    return raw_model


def _sample(raw_model, enc, cfg: TrainConfig, runtime: Runtime):
    """Sample from the raw module: the compiled one recompiles per sequence length,
    and DDP should not be driven outside the training step."""
    tokens = torch.tensor(enc.encode(cfg.sample_prompt), dtype=torch.long)
    idx = tokens.unsqueeze(0).repeat(cfg.sample_sequences, 1).to(runtime.device)
    out = raw_model.generate(idx, max_new_tokens=cfg.sample_length - idx.size(1),
                             vocab_limit=cfg.tokenizer_vocab)
    for row in out:
        print('>', enc.decode(row.tolist()))


def main(cfg: TrainConfig = None):
    train(cfg or TrainConfig())


if __name__ == '__main__':
    main()
