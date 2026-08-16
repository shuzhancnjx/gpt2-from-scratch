"""Tests for the fast (linked-list) trainer + encoder in gpt2/tokenizer.py.

BPETrainer learns merges via the O(N + merges·log N) linked-list algorithm;
BPETokenizer replays them.  There's no built-in bridge between the two, so
`build()` wires the trainer's learned tables into a fresh tokenizer.

Run with:  python3 tests/test_bpe_fast.py   (or: pytest tests/test_bpe_fast.py)
"""

import os
import sys

# bpe_base is the slow reference implementation, archived in learning/
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'learning'))

from gpt2.tokenizer import BPETrainer, BPETokenizer
from bpe_base import BPETokenizerI          # slow reference, for cross-checks


def build(text, num_merges):
    """Train with BPETrainer, return a BPETokenizer loaded with the result."""
    tr = BPETrainer()
    tr.train_bpe(text, num_merges)
    tok = BPETokenizer()
    tok.merged_pairs = tr.merged_pairs           # bridge trainer -> tokenizer
    tok.vocab = tr.vocab
    return tr, tok


def flat(tok, s):
    return [i for ids in tok.encode(s) for i in ids]


def roundtrip(tok, s):
    return tok.decode(flat(tok, s))


# ---- BPETokenizer (encoder/decoder) ---------------------------------------

def test_roundtrip_unseen_and_unicode():
    _, tok = build("the theme of these theses is thematic", 20)
    for s in ["these themes", "你好世界", "café ☕", "", " ", "x", "training hello"]:
        assert roundtrip(tok, s) == s, f"round-trip failed for {s!r}"


def test_untrained_is_identity_over_bytes():
    tok = BPETokenizer()                         # no merges loaded
    s = "hello 你好"
    assert flat(tok, s) == list(s.encode("utf-8"))
    assert roundtrip(tok, s) == s


def test_encode_is_deterministic():
    _, tok = build("banana banana banana", 10)
    assert flat(tok, "banana") == flat(tok, "banana")


def test_empty_string():
    _, tok = build("some text", 5)
    assert flat(tok, "") == []
    assert roundtrip(tok, "") == ""


# ---- BPETrainer (learned tables) ------------------------------------------

def test_merges_actually_merge():
    _, tok = build("aaaaaaaa", 5)
    assert len(flat(tok, "aaaaaaaa")) < len("aaaaaaaa".encode("utf-8"))


def test_trainer_learns_requested_number_of_merges():
    tr, _ = build("the theme of these theses is thematic", 15)
    assert len(tr.merged_pairs) == 15            # enough pairs available -> exactly 15


def test_vocab_consistent_with_merges():
    tr, _ = build("mississippi river delta", 12)
    for (a, b), idx in tr.merged_pairs.items():
        assert tr.vocab[idx] == tr.vocab[a] + tr.vocab[b], (a, b, idx)
        assert idx >= 256                        # learned ids live above the byte range


def test_more_merges_than_pairs_no_crash():
    build("ab", 100)                             # heap empties -> best_pair() None, no raise


# ---- fast vs slow cross-check (tie-break-proof) ---------------------------

def test_fast_matches_slow_behaviour():
    """Both trained on the same corpus must round-trip AND compress a held-out
    string.  We don't assert identical merge tables — tie-breaking differs
    (heap: min pair tuple; slow: first-seen) so tables may legitimately diverge."""
    corpus = "the theme of these theses is thematic and these themes theme on"
    held_out = "these thematic themes"

    _, fast = build(corpus, 30)
    slow = BPETokenizerI()
    slow.train_bpe(corpus, 30)

    assert roundtrip(fast, held_out) == held_out
    assert slow.decode([i for ids in slow.encode(held_out) for i in ids]) == held_out

    raw = len(held_out.encode("utf-8"))
    assert len(flat(fast, held_out)) < raw       # fast trainer produced useful merges


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for fn in tests:
        fn()
        print("PASS", fn.__name__)
    print(f"\nall {len(tests)} tests passed")
