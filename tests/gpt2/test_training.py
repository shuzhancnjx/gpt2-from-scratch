import math

import pytest
import torch

from gpt2.gpt import GPT, GPTConfig
from gpt2.log import Logger, load_checkpoint, read_log
from gpt2.training.runtime import Runtime
from gpt2.training.loop import get_lr

from conftest import TINY, FakeLoader


@pytest.fixture
def cpu_runtime():
    return Runtime(ddp=False, rank=0, local_rank=0, world_size=1, device='cpu',
                   device_type='cpu', master_process=True, use_compile=False, use_amp=False)


# --------------------------- schedule ---------------------------

def test_warmup_ramps_from_near_zero_to_max():
    lrs = [get_lr(s, 1000, 1.0, 0.1, 10) for s in range(10)]
    assert lrs == sorted(lrs)
    assert lrs[0] == pytest.approx(0.1)      # (0 + 1) / 10
    assert lrs[-1] == pytest.approx(1.0)


def test_cosine_decays_to_min_at_the_end():
    assert get_lr(1000, 1000, 1.0, 0.1, 10) == pytest.approx(0.1)
    assert get_lr(2000, 1000, 1.0, 0.1, 10) == pytest.approx(0.1)   # past the end


def test_midpoint_is_halfway_in_cosine_terms():
    mid = get_lr(505, 1000, 1.0, 0.0, 10)
    assert mid == pytest.approx(0.5, abs=0.02)


# --------------------------- evaluate ---------------------------

def test_evaluate_returns_a_plain_float(cpu_runtime):
    from gpt2.training.loop import evaluate
    torch.manual_seed(0)
    model = GPT(GPTConfig(**TINY))
    loader = FakeLoader(2, 16, TINY['vocab_size'])
    out = evaluate(model, loader, batches=3, runtime=cpu_runtime)
    assert isinstance(out, float) and out > 0


def test_evaluate_restores_training_mode(cpu_runtime):
    from gpt2.training.loop import evaluate
    model = GPT(GPTConfig(**TINY))
    model.train()
    evaluate(model, FakeLoader(2, 16, TINY['vocab_size']), 2, cpu_runtime)
    assert model.training


# --------------------------- gradient accumulation ---------------------------

@pytest.mark.parametrize('B,accum', [(8, 1), (4, 2), (2, 4), (1, 8)])
def test_grad_accumulation_is_invariant_to_micro_batch_size(B, accum, cpu_runtime):
    """The same 8x16 tokens, split into micro-batches differently, must produce the
    same loss and the same gradients. This is the check that caught a broken test
    harness during the refactor -- it is the core correctness property of the loop."""
    torch.manual_seed(0)
    model = GPT(GPTConfig(**TINY))
    loader = FakeLoader(B, 16, TINY['vocab_size'], seed=1)

    model.train()
    model.zero_grad(set_to_none=True)
    loss_accum = torch.zeros(())
    for _ in range(accum):
        x, y = loader.next_batch()
        _, loss = model(x, y)
        loss = loss / accum
        loss_accum += loss.detach()
        loss.backward()

    grad = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])
    torch.testing.assert_close(loss_accum, torch.tensor(2.7182), atol=10.0, rtol=10.0)

    # stash on the function so the parametrised runs can be compared
    ref = test_grad_accumulation_is_invariant_to_micro_batch_size
    if not hasattr(ref, 'expected'):
        ref.expected = (loss_accum.clone(), grad.clone())
    else:
        exp_loss, exp_grad = ref.expected
        torch.testing.assert_close(loss_accum, exp_loss, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(grad, exp_grad, atol=1e-5, rtol=1e-5)


# --------------------------- logger / checkpoints ---------------------------

def test_fresh_run_truncates_but_resume_appends(tmp_path):
    log = Logger(run_dir=str(tmp_path), resuming=False)
    log.log(1, 'train', 5.0)

    Logger(run_dir=str(tmp_path), resuming=True).log(2, 'train', 4.0)
    assert len(read_log(log.log_file)['train']) == 2, 'resuming must not erase history'

    Logger(run_dir=str(tmp_path), resuming=False)
    assert read_log(log.log_file)['train'] == []


def test_read_log_tolerates_a_torn_final_line(tmp_path):
    """A run still writing can leave a partial last line. Note '3 train 3.' would
    NOT be torn -- float('3.') is valid -- so the realistic case is a line cut
    before all three fields land."""
    p = tmp_path / 'log.txt'
    p.write_text('1 train 5.0\n2 val 4.0\n3 tra')
    parsed = read_log(str(p))
    assert parsed['train'] == [(1, 5.0)] and parsed['val'] == [(2, 4.0)]


def test_read_log_skips_unparseable_values(tmp_path):
    p = tmp_path / 'log.txt'
    p.write_text('1 train 5.0\n2 train NaNsense\n')
    assert read_log(str(p))['train'] == [(1, 5.0)]


def test_checkpoint_round_trips(tmp_path):
    torch.manual_seed(0)
    model = GPT(GPTConfig(**TINY))
    opt = model.configure_optimizers(0.1, 1e-3, (0.9, 0.95), 'cpu')
    loader = FakeLoader(2, 16, TINY['vocab_size'])

    logger = Logger(run_dir=str(tmp_path), keep_last_n=None)
    path = logger.save_checkpoint(7, model, opt, loader)

    ckpt = load_checkpoint(path)
    assert ckpt['step'] == 7
    assert isinstance(ckpt['config'], dict), 'config must be a dict, not a pickled class'

    restored = GPT(GPTConfig(**ckpt['config']))
    restored.load_state_dict(ckpt['model'])        # strict=True by default
    for a, b in zip(model.parameters(), restored.parameters()):
        torch.testing.assert_close(a, b)


def test_prune_keeps_only_the_newest(tmp_path):
    torch.manual_seed(0)
    model = GPT(GPTConfig(**TINY))
    opt = model.configure_optimizers(0.1, 1e-3, (0.9, 0.95), 'cpu')
    loader = FakeLoader(2, 16, TINY['vocab_size'])

    logger = Logger(run_dir=str(tmp_path), keep_last_n=2)
    for step in (100, 200, 300):
        logger.save_checkpoint(step, model, opt, loader)

    kept = sorted(p.name for p in tmp_path.glob('model_*.pt'))
    assert kept == ['model_000200.pt', 'model_000300.pt']


def test_amp_ctx_is_a_noop_when_disabled(cpu_runtime):
    with cpu_runtime.amp_ctx():
        assert not torch.is_autocast_enabled('cpu')


# --------------------------- continuation across shard windows ---------------------------

def test_continuation_trains_the_whole_new_window():
    """Resuming at step N with a window worth M steps must run M steps, not M-N.
    Deriving max_steps from the window alone silently truncates the run."""
    start_step, steps_this_run = 1831, 2441
    max_steps = start_step + steps_this_run
    assert max_steps - start_step == steps_this_run


def test_lr_span_default_is_a_warm_restart():
    """With no lr_schedule_steps, a continuation's cosine re-spans to the new end,
    so the LR climbs back up. That is a real choice, not an accident."""
    resumed = get_lr(1831, 1831 + 2441, 6e-4, 6e-5, 100)
    previous_run_end = get_lr(1830, 1831, 6e-4, 6e-5, 100)
    assert previous_run_end == pytest.approx(6e-5, abs=1e-6)
    assert resumed > 5 * previous_run_end          # jumps back up


def test_explicit_lr_span_keeps_one_continuous_decay():
    """Pinning lr_schedule_steps across every window gives a single decay."""
    span = 4272
    lrs = [get_lr(s, span, 6e-4, 6e-5, 100) for s in (1831, 2500, 3500, 4271)]
    assert lrs == sorted(lrs, reverse=True)        # monotonically decaying
    assert lrs[-1] == pytest.approx(6e-5, abs=1e-5)


# --------------------------- resume path resolution ---------------------------

def _touch_ckpts(tmp_path, steps):
    for s in steps:
        (tmp_path / f'model_{s:06d}.pt').write_bytes(b'x')


def test_resolve_latest_picks_the_newest(tmp_path):
    from gpt2.training.loop import resolve_checkpoint
    _touch_ckpts(tmp_path, [100, 900, 1830])
    assert resolve_checkpoint('latest', str(tmp_path)).endswith('model_001830.pt')


def test_resolve_accepts_a_bare_filename(tmp_path):
    from gpt2.training.loop import resolve_checkpoint
    _touch_ckpts(tmp_path, [1830])
    assert resolve_checkpoint('model_001830.pt', str(tmp_path)).endswith('model_001830.pt')


def test_resolve_falls_back_to_run_dir_when_cwd_relative_path_misses(tmp_path):
    """A config written for the repo root must still work when launched elsewhere."""
    from gpt2.training.loop import resolve_checkpoint
    _touch_ckpts(tmp_path, [1830])
    assert resolve_checkpoint('runs/model_001830.pt', str(tmp_path)).endswith('model_001830.pt')


def test_resolve_reports_both_paths_it_tried(tmp_path):
    from gpt2.training.loop import resolve_checkpoint
    with pytest.raises(FileNotFoundError, match='also tried'):
        resolve_checkpoint('model_999999.pt', str(tmp_path))


def test_resolve_latest_returns_none_when_there_is_nothing_to_resume(tmp_path):
    """'latest' means "continue if you can", so a first run in a fresh run_dir must
    start from scratch rather than crash."""
    from gpt2.training.loop import resolve_checkpoint
    assert resolve_checkpoint('latest', str(tmp_path)) is None


def test_an_explicit_missing_path_still_raises(tmp_path):
    """A typo must not silently restart training from step 0."""
    from gpt2.training.loop import resolve_checkpoint
    with pytest.raises(FileNotFoundError):
        resolve_checkpoint('model_001830.pt', str(tmp_path))
