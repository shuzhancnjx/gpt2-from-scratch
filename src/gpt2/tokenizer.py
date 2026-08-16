import regex 
from collections import Counter, defaultdict
import heapq

class BPETokenizer: 

    CODEX = "utf-8"

    PATTERN = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"

    def __init__(self):

        self.merged_pairs: dict[tuple[int, int], int] = {}
        self.vocab: dict[int, bytes] = {x: bytes([x]) for x in range(256)}
    

    def encode(self, text):

        tokens = regex.findall(self.PATTERN, text)

        seqs = [list(map(int, token.encode(self.CODEX))) for token in tokens]
        out = []

        for seq in seqs:

            while len(seq) >= 2:

                pairs = zip(seq, seq[1:])

                candidates = [pair for pair in pairs if pair in self.merged_pairs]

                if not candidates:
                    break 
                    
                selected_pair = min(candidates, key=self.merged_pairs.get)

                seq = self.merge(seq, selected_pair, self.merged_pairs[selected_pair])
            
            out.append(seq)

        return out


    @staticmethod
    def merge(seq, pair, next_index):

        merged, i = [], 0

        while i < len(seq):

            if i + 1 < len(seq) and (seq[i], seq[i+1]) == pair:
                merged.append(next_index)
                i+=2 
            else:
                merged.append(seq[i])
                i+=1 
        return merged 

    
    def decode(self, ids):

        return b"".join([self.vocab[id] for id in ids]).decode(self.CODEX)

class BPETrainer:
    
    CODEX = "utf-8"
    PATTERN = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"

    def __init__(self):

        self.merged_pairs: dict[tuple[int, int], int] = {}
        self.vocab: dict[int, bytes] = {x: bytes([x]) for x in range(256)}

        self.next_index = 256 

        self.freq = Counter()
        self.pair_pos = defaultdict(lambda: defaultdict(set))
        self.nxt, self.pre, self.live = [], [], []
        self.heap = []

    def train_bpe(self, text, num_merges):

        tokens = regex.findall(self.PATTERN, text)

        self.seqs = [list(map(int, token.encode(self.CODEX))) for token in tokens]

        self.build_linkedlist(self.seqs)
        self.heapify_freq()

        for _ in range(num_merges):
            best_pair = self.best_pair() 
            if best_pair: 

                self.merged_pairs[best_pair] = self.next_index
                self.vocab[self.next_index] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
                self.merge(best_pair, self.next_index, self.seqs)
                self.next_index += 1
    
    def best_pair(self):

        while self.heap:

            fre, pair = heapq.heappop(self.heap,)
            if self.freq[pair] + fre == 0 and self.freq[pair] > 0:
                return pair
        
        return None 

    def _remove(self, pair, token_pos, pos): 

        self.freq[pair] -= 1
        self.pair_pos[pair][token_pos].discard(pos)
    
    def _add(self, pair, token_pos, pos):

        self.freq[pair] += 1
        self.pair_pos[pair][token_pos].add(pos)
        
        heapq.heappush(self.heap,  (-self.freq[pair], pair))

    def merge(self, pair, next_index, seqs):

         x, y = pair 

         for i in self.pair_pos[pair]:
             
             next = self.nxt[i]
             prev = self.pre[i]
             alive = self.live[i]

             tokens = seqs[i]

             for c in list(self.pair_pos[pair][i]):
                 
                n = next[c]

                if not alive[c] or n == -1 or tokens[c] != x or tokens[n] != y:
                    continue


                p, q = prev[c], next[n]

                if p != -1: self._remove((tokens[p], tokens[c]), i, p)
                if q != -1: self._remove((tokens[n], tokens[q]), i, n)

                self._remove(pair, i, c)

                tokens[c] = next_index

                next[c] = q 
                if q != -1: 
                    prev[q] = c
                alive[n] = False 

                if p!=-1: self._add((tokens[p], tokens[c]), i, p)
                if q != -1: self._add((tokens[c], tokens[q]), i, c)
        
    
    def build_linkedlist(self, seqs):

        for i, seq in enumerate(seqs):

            for j, pair in enumerate(zip(seq, seq[1:])):

                self.freq[pair] += 1
                self.pair_pos[pair][i].add(j)

            self.pre.append([k-1 for k in range(len(seq))])
            self.nxt.append([k+1 for k in range(len(seq))])
            if self.nxt[-1]:
                self.nxt[-1][-1] = -1

            self.live.append([True for k in range(len(seq))])
        
    
    def heapify_freq(self):

        for pair in self.freq:
            heapq.heappush(self.heap, (-self.freq[pair], pair))
    





