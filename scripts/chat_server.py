#!/usr/bin/env python
"""A local chat UI and OpenAI-compatible API over the paged engine (FastAPI + uvicorn).

    python scripts/chat_server.py                     # then open http://127.0.0.1:8765
    python scripts/chat_server.py --device cpu --max-new 48 --temperature 0
    python scripts/chat_server.py --cuda-graph        # CUDA: replay decode as a graph

    curl -s localhost:8765/v1/completions -H 'Content-Type: application/json' \\
         -d '{"prompt": "The capital of France is", "max_tokens": 16, "temperature": 0}'

API docs at http://127.0.0.1:8765/docs. The server itself is gpt2's app/chat_server.py;
this script only parses flags, builds the LLM and starts uvicorn.

Logs: what the engine is doing, as it does it --

    python scripts/chat_server.py --log-level DEBUG    # + every decode step and full prompts
    tail -f results/chat_server.log                    # follow from another terminal
"""

import argparse
import logging
from pathlib import Path

import uvicorn

from app import LLM, SamplingParams
from app.chat_server import create_app

log = logging.getLogger('chat_server')

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CKPT = REPO_ROOT / 'runs/keep/gpt2_124M_step013147.pt'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', default=str(DEFAULT_CKPT), help="a checkpoint path, or 'gpt2'")
    p.add_argument('--device', default='auto')
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', type=int, default=8765)
    p.add_argument('--max-seqs', type=int, default=8)
    p.add_argument('--block-size', type=int, default=16)
    p.add_argument('--max-new', type=int, default=64)
    p.add_argument('--temperature', type=float, default=0.8)
    p.add_argument('--top-k', type=int, default=40)
    p.add_argument('--cuda-graph', action='store_true', help='CUDA: replay the decode step as a graph')
    p.add_argument('--history-turns', type=int, default=0,
                   help='earlier exchanges to include in the prompt (0 = single-turn; '
                        'a base model copies its own past replies)')
    p.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING'])
    p.add_argument('--log-file', default=str(REPO_ROOT / 'results/chat_server.log'),
                   help="also write logs here ('' to disable)")
    args = p.parse_args()

    handlers = [logging.StreamHandler()]
    if args.log_file:
        Path(args.log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(args.log_file))
    logging.basicConfig(level=args.log_level, handlers=handlers,
                        format='%(asctime)s %(levelname)-5s %(name)s | %(message)s',
                        datefmt='%H:%M:%S')

    log.info('loading %s …', args.checkpoint)
    llm = LLM(args.checkpoint, engine='paged', device=args.device, max_seqs=args.max_seqs,
              block_size=args.block_size, cuda_graph=args.cuda_graph)
    sampling_params = SamplingParams(max_new_tokens=args.max_new, temperature=args.temperature,
                            top_k=args.top_k)
    llm.generate(['Hello'], SamplingParams(max_new_tokens=2, temperature=sampling_params.temperature,
                                           top_k=sampling_params.top_k))            # warm-up (+ graph)

    info = dict(checkpoint=Path(args.checkpoint).name, block_size=args.block_size,
                num_blocks=llm.engine.blocks.allocator.num_blocks)
    app = create_app(llm, sampling_params, history_turns=args.history_turns, info=info)
    log.info('ready: http://%s:%d  (API docs at /docs, log level %s%s)', args.host, args.port,
             args.log_level, f', also -> {args.log_file}' if args.log_file else '')
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level.lower(),
                access_log=args.log_level == 'DEBUG')


if __name__ == '__main__':
    main()
