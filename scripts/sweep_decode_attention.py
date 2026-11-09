"""Decode attention vs its bandwidth floor: tile size x rows x context length (CUDA).

    python scripts/sweep_decode_attention.py
    python scripts/sweep_decode_attention.py --rows 16 64 --ctx 256 1024 --tiles 32 64

The decode kernel (paged/attention/decode_attention.py) must read every row's K and V:
rows x ctx x heads x head_dim x 2 (K, V) x 2 bytes. At the GPU's copy bandwidth that is
its floor. Each cell is the kernel alone, replayed from a CUDA graph (no launch cost),
reported as microseconds and as % of that floor:

    low %  -> the kernel is WAITING: too few programs (rows x heads) with too long a chain
              of tiles each -- bigger tiles or split-KV help
    high % -> it is moving bytes as fast as the memory allows -- only fewer bytes help

One layer (12 heads, head_dim 64, block size 16, fp16), GPT-2 124M's shapes.
"""

import argparse

import torch

from inference.continuous_batch.paged.attention.decode_attention import decode_attention


def make(rows, ctx, heads=12, head_dim=64, block_size=16, dtype=torch.float16):
    """A paged pool with `rows` requests of `ctx` tokens each, blocks scattered."""
    blocks_per_row = -(-ctx // block_size)
    num_blocks = rows * blocks_per_row + 1                      # + scratch block 0
    table = (torch.randperm(num_blocks - 1, device='cuda') + 1).view(rows, blocks_per_row)
    seq_lens = torch.full((rows,), ctx, device='cuda')
    new = seq_lens - 1                                          # the new token's position
    slots = table[torch.arange(rows, device='cuda'), new // block_size] * block_size + new % block_size
    qkv = torch.randn(rows, 3, heads, head_dim, device='cuda', dtype=dtype)
    pool = lambda: torch.randn(num_blocks, block_size, heads, head_dim, device='cuda', dtype=dtype)
    args = dict(query=qkv[:, 0], key=qkv[:, 1], value=qkv[:, 2], key_pool=pool(), value_pool=pool(),
                block_table=table, seq_lens=seq_lens, slots=slots)
    kv_bytes = rows * ctx * heads * head_dim * 2 * torch.finfo(dtype).bits // 8
    return args, kv_bytes


def graph_us(fn, reps=200):
    """GPU time of fn, captured in a CUDA graph and replayed (no launch overhead)."""
    for _ in range(3):
        fn()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    graph.replay()
    torch.cuda.synchronize()
    start.record()
    for _ in range(reps):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / reps * 1e3


def copy_bandwidth():
    """What this GPU's memory actually delivers: a large device-to-device copy (bytes/us)."""
    x = torch.empty(256 * 2**20, dtype=torch.uint8, device='cuda')
    y = torch.empty_like(x)
    us = graph_us(lambda: y.copy_(x), reps=50)
    return 2 * x.numel() / us                                   # read + write


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--rows', type=int, nargs='+', default=[1, 16, 64])
    p.add_argument('--ctx', type=int, nargs='+', default=[256, 1024, 4096])
    p.add_argument('--tiles', type=int, nargs='+', default=[16, 32, 64, 128, 256])
    args = p.parse_args()

    torch.manual_seed(0)
    bw = copy_bandwidth()
    print(f'{torch.cuda.get_device_name()}: copy bandwidth {bw / 1e6:.2f} TB/s '
          f'(the floor below is KV bytes / this)\n')
    tile_cols = ''.join(f'{f"tile {t}":>17s}' for t in args.tiles)
    print(f'{"rows":>5s} {"ctx":>5s} {"programs":>8s} {"KV MB":>7s} {"floor us":>8s}{tile_cols}')
    for rows in args.rows:
        for ctx in args.ctx:
            kernel_args, kv_bytes = make(rows, ctx)
            floor = kv_bytes / bw
            cells = []
            for tile in args.tiles:
                us = graph_us(lambda: decode_attention(**kernel_args, tile=tile))
                cells.append(f'{us:8.1f} us {floor / us:4.0%}')
            print(f'{rows:5d} {ctx:5d} {rows * 12:8d} {kv_bytes / 2**20:7.1f} {floor:8.1f}'
                  + ''.join(f'{c:>17s}' for c in cells))
        print()


if __name__ == '__main__':
    main()
