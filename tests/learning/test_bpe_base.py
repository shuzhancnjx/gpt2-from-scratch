"""Tests for BPETokenizerI.

Run with:  python3 test_bpe.py
(no pytest needed — plain asserts + a tiny runner)

`encode` returns a list-of-lists (one list per pre-tokenized piece), so we
flatten it before decoding.  `flat` handles the [] / "" empty cases too.
"""

import os, sys
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'learning'))
from bpe_base import BPETokenizerI


def flat(tok, s):
    """encode -> flat list[int]."""
    return [i for ids in tok.encode(s) for i in ids]


def roundtrip(tok, s):
    return tok.decode(flat(tok, s))


def test_roundtrip_unseen_and_unicode():
    t = BPETokenizerI()
    t.train_bpe("the theme of these theses is thematic", 20)
    for s in ["these themes", "你好世界", "café ☕", "", " ", "x", "training hello"]:
        assert roundtrip(t, s) == s, f"round-trip failed for {s!r}"


def test_untrained_is_identity_over_bytes():
    t = BPETokenizerI()                       # no merges learned
    s = "hello 你好"
    assert flat(t, s) == list(s.encode("utf-8"))   # relies on lossless pre-tokenization
    assert roundtrip(t, s) == s


def test_encode_is_deterministic():
    t = BPETokenizerI()
    t.train_bpe("banana banana banana", 10)
    assert flat(t, "banana") == flat(t, "banana")


def test_merges_actually_merge():
    t = BPETokenizerI()
    t.train_bpe("aaaaaaaa", 5)
    assert len(flat(t, "aaaaaaaa")) < len("aaaaaaaa".encode("utf-8"))


def test_vocab_consistent_with_merges():
    t = BPETokenizerI()
    t.train_bpe("mississippi river", 10)
    for (a, b), idx in t.pair_merge.items():
        assert t.vocab[idx] == t.vocab[a] + t.vocab[b], (a, b, idx)


def test_more_merges_than_pairs_no_crash():
    t = BPETokenizerI()
    t.train_bpe("ab", 100)                     # must not raise (None-guard)


def test_empty_string():
    t = BPETokenizerI()
    t.train_bpe("some text", 5)
    assert flat(t, "") == []
    assert roundtrip(t, "") == ""


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for fn in tests:
        fn()
        print("PASS", fn.__name__)
    print(f"\nall {len(tests)} tests passed")
