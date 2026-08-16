# BPE Tokenizer — Review Notes

Review of `bpe.py` (byte-pair encoding tokenizer with GPT-2 style pre-tokenization).

## Critical bugs

### 1. `encode` is broken three ways (bpe.py:22-46)
- **Wrong selection criterion.** It picks `max_frequencey_pair()` — the most *frequent* pair in the new text. Encoding must pick the pair with the **lowest merge id** (the merge learned earliest = highest priority), so tokenization stays consistent with training.
- **Stale loop variable → infinite loop.** `for indices in self.indices` binds `indices` to a list, but `self.merge(...)` reassigns `self.indices[j] = merged`. The local `indices` still points at the old, unmerged list, so `while len(indices) >= 2` and `candidates` never change while the pair is already gone from `self.indices[j]` — the loop never exits.
- **Mutates training state.** `encode` calls the stateful `self.merge`, corrupting `self.indices` and `self.frequency`. Encoding must be pure.

Fix — pure, id-ranked encode:

```python
def encode(self, text):
    out = []
    for piece in regex.findall(self.PATTERN, text):
        seq = list(piece.encode(self.CODEX))
        while len(seq) >= 2:
            pairs = set(zip(seq, seq[1:]))
            candidates = [p for p in pairs if p in self.pair_merge]
            if not candidates:
                break
            pair = min(candidates, key=lambda p: self.pair_merge[p])   # earliest-learned wins
            seq = self._merge_seq(seq, pair, self.pair_merge[pair])
        out.extend(seq)                # flat list[int]
    return out

@staticmethod
def _merge_seq(seq, pair, new_index):
    merged, i = [], 0
    while i < len(seq):
        if i + 1 < len(seq) and (seq[i], seq[i+1]) == pair:
            merged.append(new_index); i += 2
        else:
            merged.append(seq[i]); i += 1
    return merged
```

### 2. Incremental frequency update is incorrect for adjacent merges (bpe.py:87-94)
When two merges happen back-to-back, the left-neighbor update reads the *old* token (`indices[i-1]`) instead of the newly-merged `new_index`. Trace `[A,B,A,B]` merging `(A,B)→X`: the true result `[X,X]` should give `{(X,X): 1}`, but the code produces `{(X,A):1, (B,X):1}` and misses `(X,X)`. The counter drifts and later iterations pick wrong merges.

Fix — simplest correct option is to **recompute** the counts each iteration in `train_bpe` instead of the incremental updates in `merge`. Keep the incremental approach only if you switch to a linked-list structure.

### 3. Crash: `train_bpe` when no pair is left (bpe.py:68)
When `max_frequencey_pair()` returns `None`, `self.pair_merge[None]` raises `KeyError`. Break on `None`:

```python
pair = self.max_frequencey_pair()
if pair is None:
    break
```

## Minor
- Typos in the public API: `decoode`, `max_frequencey_pair`, `selelcted_pair`.
- `tuple(int, int)` → `tuple[int, int]` (bpe.py:15, bpe.py:20) — subscript, not a call.
- Unused `import string`.

## Still missing (to be a complete tokenizer)
- `save` / `load` — serialize `pair_merge` + `vocab` to disk so you don't retrain each run.
- Special tokens (e.g. `<|endoftext|>`).
- `decode` robustness for invalid/partial id sequences (`errors="replace"` vs strict).
- Deterministic tie-breaking in `max` during training for full reproducibility.

## Tests to add
Dependency-free (plain asserts, run with `python3 test_bpe.py`). Assumes `encode` returns a flat `list[int]`.

```python
from bpe import BPETokenizer

def test_roundtrip_unseen_and_unicode():
    t = BPETokenizer()
    t.train_bpe("the theme of these theses is thematic", 20)
    for s in ["these themes", "你好世界", "café ☕", "", " ", "x"]:
        assert t.decode(t.encode(s)) == s, s

def test_untrained_is_identity_over_bytes():
    t = BPETokenizer()                      # no merges learned
    s = "hello 你好"
    assert t.encode(s) == list(s.encode("utf-8"))
    assert t.decode(t.encode(s)) == s

def test_encode_is_deterministic():
    t = BPETokenizer(); t.train_bpe("banana banana", 10)
    assert t.encode("banana") == t.encode("banana")

def test_merges_actually_merge():
    t = BPETokenizer(); t.train_bpe("aaaa", 5)
    assert len(t.encode("aaaa")) < len("aaaa".encode("utf-8"))

def test_vocab_consistent_with_merges():
    t = BPETokenizer(); t.train_bpe("mississippi", 10)
    for (a, b), idx in t.pair_merge.items():
        assert t.vocab[idx] == t.vocab[a] + t.vocab[b]

def test_more_merges_than_pairs_no_crash():
    t = BPETokenizer(); t.train_bpe("ab", 100)   # must not raise

if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print("PASS", name)
    print("all passed")
```

What each test guards against:
- `test_roundtrip_*` — the encode selection/loop bug and any decode mismatch.
- `test_untrained_is_identity` — the 256-byte base behavior.
- `test_encode_is_deterministic` — a non-pure encode mutating shared state.
- `test_merges_actually_merge` — merges take effect and compression happens.
- `test_vocab_consistent_with_merges` — the invariant `vocab[id] == vocab[a] + vocab[b]`.
- `test_more_merges_than_pairs_no_crash` — the `None`/empty-counts crash.

## Optimization: incremental frequency updates via a linked list

The recompute-each-round trainer above is correct and simple, but O(N · merges). The fast version keeps `freq` up to date incrementally. The array-rewrite attempt drifts because after a merge, positions shift and "neighbor" reads the pre-merge value. A **doubly linked list** fixes both: neighbors are always the current live tokens, and splicing a node out is O(1).

### A merge is a local edit — only 5 counts change
Given context `… P A B Q …`, merging `(A,B)` → `X`:

```
before:   P — A — B — Q          after:   P — X — Q
destroyed: (P,A) (A,B) (B,Q)
created:   (P,X) (X,Q)

freq[(P,A)] -= 1     freq[(P,X)] += 1
freq[(A,B)] -= 1     freq[(X,Q)] += 1
freq[(B,Q)] -= 1
```

This is correct (unlike the array version) because `P` and `Q` are read from the list, so they reflect earlier merges. The array code read `indices[i-1]`, the pre-merge value — that was the drift bug.

### Representation: index-based linked list
Parallel arrays over the initial byte sequence; nodes are marked dead, never deleted:

```python
tokens = list(text.encode("utf-8"))
n = len(tokens)
prev = [i - 1 for i in range(n)]      # -1 at start
next = [i + 1 for i in range(n)]; next[-1] = -1
alive = [True] * n
```

Splice node `j` (the `B`) out after merging into node `i` (now `X`):

```python
tokens[i] = X
q = next[j]; next[i] = q
if q != -1: prev[q] = i
alive[j] = False
```

### Merge all occurrences of a pair in one round
Keep a reverse index `pair_pos: dict[pair, set[int]]` of left-node positions, built alongside `freq`:

```python
def _add(pr, pos):    freq[pr] += 1; pair_pos[pr].add(pos)
def _remove(pr, pos): freq[pr] -= 1; pair_pos[pr].discard(pos)

def merge_round(pair):                    # pair = (a, b) -> new id X
    a, b = pair
    for i in list(pair_pos[pair]):        # snapshot — set mutates during loop
        j = next[i]
        # re-validate: an earlier merge this round may have invalidated this spot
        if not alive[i] or j == -1 or tokens[i] != a or tokens[j] != b:
            continue
        p, q = prev[i], next[j]

        if p != -1: _remove((tokens[p], a), p)   # destroyed
        if q != -1: _remove((b, tokens[q]), j)
        _remove(pair, i)

        tokens[i] = X                            # splice
        next[i] = q
        if q != -1: prev[q] = i
        alive[j] = False

        if p != -1: _add((tokens[p], X), p)      # created
        if q != -1: _add((X, tokens[q]), i)
```

### The overlap gotcha (why re-validation matters)
Repeated symbols, e.g. `AAAA` merging `(A,A)`: snapshot positions `{0,1,2}`. Merging occurrence 0 (nodes 0–1) writes `X` at 0, splices node 1, and its right-pair removal discards position 1 from `pair_pos[(A,A)]`. When the loop reaches `i=1` the `alive`/`tokens` check skips it. Result: `[X, X]`, `freq = {(X,X): 1}` — correct, where the array version produced garbage. The re-validation line is what makes overlaps safe.

### Fast best-pair selection: lazy-deletion max-heap
Recomputing `max(freq)` each round is O(#unique pairs). Instead push on every count change and skip stale entries on pop:

```python
import heapq
# on every _add/_remove: heapq.heappush(heap, (-freq[pr], pr))
def best_pair():
    while heap:
        negc, pr = heapq.heappop(heap)
        if freq.get(pr, 0) == -negc and freq[pr] > 0:
            return pr
    return None
```

### Payoff
- Correctness: neighbors are always current → no drift.
- Complexity: ~O(N + merges · log N) vs O(N · merges).
- In practice this runs *within a single pre-tokenized piece* (pairs never cross word boundaries), so you run it per unique word weighted by frequency, shrinking N enormously.

Reach for this only when training speed actually bites — the recompute version is fine for learning.
