"""Tokenize FineWeb-Edu (sample-10BT) into raw uint16 shards of 10M tokens each.

    python scripts/prepare_data.py                         # everything, original layout
    python scripts/prepare_data.py --out-dir /content/edu_fineweb10B --max-shards 251

Shard 0 is validation, shards 1.. are training. --max-shards stops early and streams the
dataset instead of downloading it, so a 2.5B-token run (1 val + 250 train shards) never
fetches the full 10B tokens.
"""

import argparse
import multiprocessing as mp
import os

import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

remote_name = "sample-10BT"
shard_size = int(1e7)
DEFAULT_OUT = os.path.join(os.path.dirname(__file__), "edu_fineweb10B")

enc = tiktoken.get_encoding('gpt2')
eot = enc._special_tokens['<|endoftext|>']


def tokenize(doc):
    tokens = [eot]
    tokens.extend(enc.encode_ordinary(doc["text"]))
    return np.array(tokens).astype(np.uint16)


def write_datafile(filename, tokens_np):
    with open(filename, 'wb') as f:
        f.write(tokens_np.tobytes())


def shard_path(out_dir, shard_index):
    split = 'val' if shard_index == 0 else 'train'
    return os.path.join(out_dir, f"edufineweb_{split}_{shard_index:06d}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out-dir', default=DEFAULT_OUT)
    p.add_argument('--max-shards', type=int, default=None,
                   help='stop after this many shards (1 val + N-1 train); streams the dataset')
    p.add_argument('--shard-size', type=int, default=shard_size,
                   help='tokens per shard (default 10M; smaller only for smoke tests)')
    args = p.parse_args()
    size = args.shard_size

    os.makedirs(args.out_dir, exist_ok=True)
    streaming = args.max_shards is not None
    fw = load_dataset("HuggingFaceFW/fineweb-edu", name=remote_name, split="train",
                      streaming=streaming)

    nprocs = max(1, os.cpu_count() // 2)
    print(f'=> number of process is {nprocs}; writing to {args.out_dir}'
          + (f', stopping after {args.max_shards} shards' if streaming else ''))

    with mp.Pool(nprocs) as pool:
        shard_index = 0
        all_tokens_np = np.empty((size,), dtype=np.uint16)
        token_count = 0
        progress_bar = None
        for tokens in pool.imap(tokenize, fw, chunksize=16):

            if token_count + len(tokens) <= size:
                all_tokens_np[token_count:token_count + len(tokens)] = tokens
                token_count += len(tokens)

                if progress_bar is None:
                    progress_bar = tqdm(total=size, unit="tokens", desc=f"Shard {shard_index}")
                progress_bar.update(len(tokens))
            else:
                remainder = size - token_count
                progress_bar.update(remainder)
                all_tokens_np[token_count:token_count + remainder] = tokens[:remainder]
                write_datafile(shard_path(args.out_dir, shard_index), all_tokens_np)
                shard_index += 1
                progress_bar.close()
                progress_bar = None
                if streaming and shard_index >= args.max_shards:
                    print(f'=> wrote {shard_index} shards, stopping')
                    return
                all_tokens_np[0:len(tokens) - remainder] = tokens[remainder:]
                token_count = len(tokens) - remainder

        if token_count != 0:
            write_datafile(shard_path(args.out_dir, shard_index), all_tokens_np[:token_count])


if __name__ == "__main__":
    main()
