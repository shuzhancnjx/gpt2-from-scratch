import collections 
import regex


class BPETokenizerI: 

    CODEX = "utf-8"
    PATTERN = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"

    def __init__(self): 

        self.indices = []
        self.pair_merge: dict[tuple[int, int], int] = {}
        self.vocab: dict[int, bytes] = { x: bytes([x]) for x in range(256)}

        self.next_index = 256

        self.frequency: dict[tuple[bytes, bytes], int] = collections.Counter()
    
    def encode(self, text): 

        if not text:
            return []

        indices = self.split_string(text)

        raw_token_n = sum([len(indice) for indice in indices])

        out = []

        for indice in indices: 

            while len(indice) >= 2: 

                pairs = zip(indice, indice[1:])

                candidates = [p for p in pairs if p in self.pair_merge]

                if not candidates:
                    break 

                selected_pair = min(candidates, key=self.pair_merge.get)

                indice = self.merge(indice, selected_pair, self.pair_merge[selected_pair])
            
            out.append(indice)

        total_token = sum([len(seq) for seq in out])
        print(f"{raw_token_n/total_token:.2} bytes/token, {total_token} token from {raw_token_n} bytes")
        return out 

    def decode(self, ids): 

        return b"".join([self.vocab[x] for x in ids]).decode(self.CODEX)
    
    def train_bpe(self, string: str,  num_merges: int):

        indices = self.split_string(string)

        for _ in range(num_merges): 

            self.count_adjacent_pairs(indices)

            pair = self.max_frequence_pair()

            if not pair:
                break

            if pair not in self.pair_merge: 

                self.pair_merge[pair] = self.next_index

                self.vocab[self.next_index] = self.vocab[pair[0]] + self.vocab[pair[1]]
                self.next_index += 1

            for j in range(len(indices)):
                indices[j] = self.merge(indices[j], pair, self.pair_merge[pair])

        return 
    

    @staticmethod
    def merge(seq, pair, new_index):
        
        merged, i = [],  0
        while i < len(seq): 

            if i + 1 < len(seq) and (seq[i], seq[i+1]) == pair: 

                merged.append(new_index)
                i += 2

            else: 

                merged.append(seq[i])
                i += 1

        return merged

    def split_string(self, text):

        indices = []
        pieces = regex.findall(self.PATTERN, text)

        for piece in pieces:
            indice = list(map(int, piece.encode(self.CODEX))) 
            indices.append(indice)
        return indices 


    def count_adjacent_pairs(self, seqs): 

        self.frequency = collections.Counter()
        for seq in seqs:
            self.frequency +=  collections.Counter([(x, y) for x, y in zip(seq, seq[1:])])
    
    def max_frequence_pair(self):

        if not self.frequency:
            return 

        return max(self.frequency, key=self.frequency.get)


if __name__ == "__main__": 

    tokenzier = BPETokenizerI() 
    tokenzier.train_bpe("hello, okay, just for a testing, right? training bpe", 10)

    print(tokenzier.decode([id for ids in tokenzier.encode("training hello") for id in ids]))

    # print(tokenzier.indices)
    # print(tokenzier.pair_merge)