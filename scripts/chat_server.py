#!/usr/bin/env python
"""A local chat UI over the paged continuous-batching engine.

    python scripts/chat_server.py                     # then open http://127.0.0.1:8765
    python scripts/chat_server.py --device cpu --max-new 48 --temperature 0

Standard library only: ThreadingHTTPServer serves one HTML page and a JSON API,
bound to localhost. Each HTTP request runs on its own thread and hands its prompt
to a single ENGINE thread -- the one writer that owns the scheduler and the KV pool
(request-lifecycle.md §20).

Batching here is *between runs*: PagedScheduler.run() takes a fixed list of
prompts, so the engine gathers whatever requests arrived close together and runs
them as one batch; a request arriving mid-run waits for the next run. Admitting
requests into a run that is already going needs a step-level engine API -- a later
milestone. The UI shows each reply's batch size so the batching is visible.

The model is a base GPT-2, not a chat model: the conversation is rendered as a
"User: / Assistant:" transcript and the reply is cut at the next turn. Expect
web-text continuations, not instruction-following.

Logs: what the engine is doing, as it does it --

    python scripts/chat_server.py                      # INFO: requests, batches, admit/retire
    python scripts/chat_server.py --log-level DEBUG    # + every decode step and full prompts
    tail -f results/chat_server.log                    # follow from another terminal
"""

import argparse
import json
import logging
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import tiktoken
import torch

from gpt2.device import default_dtype, pick_device
from gpt2.gpt import GPT, GPTConfig
from gpt2.inference.continuous_batch import PagedScheduler

log = logging.getLogger('chat_server')

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CKPT = REPO_ROOT / 'runs/keep/gpt2_124M_step013147.pt'

# Plain transcript, no few-shot examples: tried on this 124M base model, examples
# were copied into answers ('the capital of France…') rather than imitated as a
# style. A base model has never seen question-answering -- that needs SFT
# (learning plan, Phase 4).
PREAMBLE = 'Below is a conversation between a curious user and a knowledgeable assistant.\n\n'
# A base model rambles on past its answer; one line reads as one chat turn.
STOPS = ('\n', '<|endoftext|>')


# -- model and prompt ------------------------------------------------------------

def load_model(path, device):
    ck = torch.load(path, map_location='cpu', weights_only=True)
    cfg = GPTConfig(**ck['config'])
    model = GPT(cfg)
    model.load_state_dict(ck['model'])
    return model.eval().to(device=device, dtype=default_dtype(device)), cfg


def make_sampler(temperature, top_k):
    if temperature <= 0:
        return lambda logits: logits.argmax(-1, keepdim=True)

    def sample(logits):
        logits = logits.float() / temperature
        kth = torch.topk(logits, top_k, dim=-1).values[:, -1:]
        logits = logits.masked_fill(logits < kth, float('-inf'))
        return torch.multinomial(torch.softmax(logits, -1), 1)
    return sample


def render(messages, enc, budget, history_turns=0):
    """Transcript ending in 'Assistant:'; oldest turns dropped to fit the token budget.

    history_turns: how many earlier user/assistant exchanges to include. Default 0
    (single-turn) because a base model copies its own earlier replies: measured on
    this checkpoint, one 'email@example.com' reply in the history makes the next reply
    start with an email 31% of the time, two make it 69% (vs 0% with no history).
    """
    if history_turns is not None:
        messages = messages[-(2 * history_turns + 1):]
    turns = [f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content'].strip()}\n"
             for m in messages if m.get('content', '').strip()]
    while turns:
        prompt = PREAMBLE + ''.join(turns) + 'Assistant:'
        if len(enc.encode(prompt)) <= budget:
            return prompt
        if len(turns) == 1:                       # one message alone is too long: keep its tail
            ids = enc.encode(turns[0])[-(budget - 64):]
            return PREAMBLE + enc.decode(ids) + 'Assistant:'
        turns.pop(0)
    return PREAMBLE + 'Assistant:'


def cut_reply(text):
    for stop in STOPS:
        i = text.find(stop)
        if i != -1:
            text = text[:i]
    return text.strip() or '…'


# -- the engine thread: the single writer ------------------------------------------

class Engine(threading.Thread):

    def __init__(self, sched, sampler, max_new, max_batch, gather_window=0.03):
        super().__init__(daemon=True)
        self.sched, self.sampler, self.max_new = sched, sampler, max_new
        self.max_batch, self.gather_window = max_batch, gather_window
        self.inbox = queue.Queue()
        self.runs = 0

    def submit(self, prompt, timeout=600):
        job = {'prompt': prompt, 'done': threading.Event()}
        self.inbox.put(job)
        if not job['done'].wait(timeout):
            raise TimeoutError('generation timed out')
        if 'error' in job:
            raise RuntimeError(job['error'])
        return job

    def run(self):
        while True:
            batch = [self.inbox.get()]               # blocks when idle: no spinning
            time.sleep(self.gather_window)           # let near-simultaneous requests join
            while len(batch) < self.max_batch:
                try:
                    batch.append(self.inbox.get_nowait())
                except queue.Empty:
                    break

            prompts = [job['prompt'] for job in batch]
            log.info('batch %d: %d request(s) -> engine', self.runs + 1, len(batch))
            t0 = time.perf_counter()
            try:
                outs = self.sched.run(prompts, max_new_tokens=self.max_new, sampler=self.sampler)
            except Exception as e:                   # report, keep serving
                for job in batch:
                    job['error'] = f'{type(e).__name__}: {e}'
                    job['done'].set()
                continue
            seconds = time.perf_counter() - t0
            self.runs += 1
            tokens = sum(len(g) for g in self.sched.generated_cpu.values())
            log.info('batch %d done in %.2fs: %d tokens (%.1f tok/s), %d decode steps, '
                     'peak %d blocks / %d requests resident',
                     self.runs, seconds, tokens, tokens / seconds, self.sched.decode_steps,
                     self.sched.peak_blocks, self.sched.peak_active)

            for i, (job, out) in enumerate(zip(batch, outs)):
                job['raw'] = out[len(job['prompt']):]
                job['reply'] = cut_reply(job['raw'])
                job['stats'] = dict(batch_size=len(batch), seconds=round(seconds, 2),
                                    tokens=len(self.sched.generated_cpu.get(i, [])),
                                    resident=self.sched.peak_active,
                                    blocks=self.sched.peak_blocks)
                job['done'].set()


# -- HTTP ------------------------------------------------------------------------

def make_handler(engine, enc, budget, info, history_turns):

    class Handler(BaseHTTPRequestHandler):

        def _send(self, code, body, ctype='application/json'):
            data = body if isinstance(body, bytes) else json.dumps(body).encode()
            self.send_response(code)
            self.send_header('Content-Type', ctype)
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path in ('/', '/index.html'):
                self._send(200, PAGE.encode(), 'text/html; charset=utf-8')
            elif self.path == '/api/info':
                self._send(200, dict(info, runs=engine.runs))
            else:
                self._send(404, {'error': 'not found'})

        def do_POST(self):
            if self.path != '/api/chat':
                return self._send(404, {'error': 'not found'})
            try:
                body = json.loads(self.rfile.read(int(self.headers.get('Content-Length', 0))))
                messages = body['messages']
                assert isinstance(messages, list) and messages
            except Exception:
                return self._send(400, {'error': 'expected {"messages": [{"role", "content"}, ...]}'})
            prompt = render(messages, enc, budget, history_turns)
            log.info('user: %s', _clip(messages[-1].get('content', '')))
            log.debug('prompt (%d tokens):\n%s', len(enc.encode(prompt)), prompt)
            try:
                job = engine.submit(prompt)
            except Exception as e:
                log.error('request failed: %s', e)
                return self._send(500, {'error': str(e)})
            log.debug('raw output (before the cut at the first newline):\n%s', job['raw'])
            log.info('reply: %s  [%s]', _clip(job['reply']),
                     ', '.join(f'{k}={v}' for k, v in job['stats'].items()))
            self._send(200, {'reply': job['reply'], 'stats': job['stats']})

        def log_message(self, fmt, *args):         # HTTP access lines: DEBUG only
            log.debug('%s %s', self.address_string(), fmt % args)

    return Handler


def _clip(text, n=100):
    text = ' '.join(str(text).split())
    return text if len(text) <= n else text[:n] + '…'


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>GPT-2 Chat</title>
<style>
  :root { --bg:#f7f7f5; --panel:#ffffff; --text:#1d1d1f; --muted:#6e6e73; --line:#e3e3e0;
          --user:#2f6fde; --user-text:#ffffff; --bot:#efefec; --accent:#2f6fde; }
  @media (prefers-color-scheme: dark) {
    :root { --bg:#161617; --panel:#1f1f21; --text:#ececee; --muted:#9a9aa0; --line:#323236;
            --user:#3a78e6; --user-text:#ffffff; --bot:#2a2a2d; --accent:#5b93f0; }
  }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--text);
         font:15px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
  .wrap { max-width:760px; margin:0 auto; padding:0 16px; height:100vh; display:flex; flex-direction:column; }
  header { padding:16px 0 10px; border-bottom:1px solid var(--line); }
  h1 { margin:0; font-size:18px; }
  .info { color:var(--muted); font-size:12.5px; margin-top:4px; }
  #log { flex:1; overflow-y:auto; padding:16px 0; display:flex; flex-direction:column; gap:10px; }
  .msg { max-width:85%; padding:9px 13px; border-radius:14px; white-space:pre-wrap; word-wrap:break-word; }
  .user { align-self:flex-end; background:var(--user); color:var(--user-text); border-bottom-right-radius:4px; }
  .bot  { align-self:flex-start; background:var(--bot); border-bottom-left-radius:4px; }
  .meta { align-self:flex-start; color:var(--muted); font-size:11.5px; margin:-6px 0 0 6px; }
  .err  { color:#d33; }
  form { display:flex; gap:8px; padding:12px 0 16px; border-top:1px solid var(--line); }
  textarea { flex:1; resize:none; height:46px; padding:11px 12px; border-radius:12px; border:1px solid var(--line);
             background:var(--panel); color:var(--text); font:inherit; }
  button { border:0; border-radius:12px; padding:0 16px; background:var(--accent); color:#fff; font:inherit; cursor:pointer; }
  button.ghost { background:transparent; color:var(--accent); border:1px solid var(--line); }
  button:disabled { opacity:.5; cursor:default; }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>GPT-2 Chat <span style="font-weight:400;color:var(--muted)">· paged engine</span></h1>
    <div class="info" id="info">loading…</div>
  </header>
  <div id="log"></div>
  <form id="f">
    <textarea id="q" placeholder="Say something… (Enter to send, Shift+Enter for a newline)"></textarea>
    <button id="send">Send</button>
    <button id="newchat" type="button" class="ghost" title="Forget this conversation">New chat</button>
    <button id="burst" type="button" class="ghost" title="Send this message as 4 separate requests at once, to see them batched">×4</button>
  </form>
</div>
<script>
const log = document.getElementById('log'), q = document.getElementById('q');
const send = document.getElementById('send'), burst = document.getElementById('burst');
const history = [];

function add(cls, text) {
  const d = document.createElement('div');
  d.className = cls; d.textContent = text;
  log.appendChild(d); log.scrollTop = log.scrollHeight;
  return d;
}

async function ask(messages) {
  const r = await fetch('/api/chat', {method:'POST', headers:{'Content-Type':'application/json'},
                                      body: JSON.stringify({messages})});
  const j = await r.json();
  if (!r.ok) throw new Error(j.error || r.status);
  return j;
}

function meta(s) {
  return `batch ${s.batch_size} · ${s.seconds}s · ${s.tokens} tokens · ${s.blocks} blocks in use`;
}

async function onSend(times) {
  const text = q.value.trim();
  if (!text) return;
  q.value = ''; send.disabled = burst.disabled = true;
  add('msg user', times > 1 ? `${text}   (×${times})` : text);
  const turn = [...history, {role:'user', content:text}];
  const pending = Array.from({length: times}, () => add('msg bot', '…'));
  const results = await Promise.allSettled(pending.map(() => ask(turn)));
  results.forEach((res, i) => {
    if (res.status === 'fulfilled') {
      pending[i].textContent = res.value.reply;
      const m = add('meta', meta(res.value.stats));
      pending[i].after(m);
    } else {
      pending[i].textContent = 'Error: ' + res.reason.message;
      pending[i].classList.add('err');
    }
  });
  const first = results.find(r => r.status === 'fulfilled');
  history.push({role:'user', content:text});
  if (first) history.push({role:'assistant', content:first.value.reply});
  send.disabled = burst.disabled = false; q.focus();
}

document.getElementById('f').addEventListener('submit', e => { e.preventDefault(); onSend(1); });
burst.addEventListener('click', () => onSend(4));
document.getElementById('newchat').addEventListener('click', () => {
  history.length = 0; log.innerHTML = ''; q.focus();
});
q.addEventListener('keydown', e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); onSend(1); } });

fetch('/api/info').then(r => r.json()).then(i => {
  document.getElementById('info').textContent =
    `${i.checkpoint} · ${i.device} · max ${i.max_seqs} concurrent · block size ${i.block_size} · ` +
    `${i.num_blocks} blocks · temperature ${i.temperature} · up to ${i.max_new} new tokens · ` +
    (i.history_turns ? `remembers ${i.history_turns} earlier exchange(s)` : 'single-turn (no memory of earlier messages)') +
    ` · base model, not chat-tuned`;
});
q.focus();
</script>
</body>
</html>
"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', default=str(DEFAULT_CKPT))
    p.add_argument('--device', default='auto')
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', type=int, default=8765)
    p.add_argument('--max-seqs', type=int, default=8)
    p.add_argument('--block-size', type=int, default=16)
    p.add_argument('--max-new', type=int, default=64)
    p.add_argument('--temperature', type=float, default=0.8)
    p.add_argument('--top-k', type=int, default=40)
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

    device = pick_device(args.device)
    log.info('loading %s on %s …', args.checkpoint, device)
    model, cfg = load_model(args.checkpoint, device)
    enc = tiktoken.get_encoding('gpt2')

    sched = PagedScheduler(model, enc, cfg, max_seqs=args.max_seqs, block_size=args.block_size,
                           max_tokens=cfg.block_size, device=device, dtype=default_dtype(device))
    sampler = make_sampler(args.temperature, args.top_k)
    sched.run(['Hello'], max_new_tokens=2, sampler=sampler)          # warm-up
    engine = Engine(sched, sampler, args.max_new, args.max_seqs)
    engine.start()

    info = dict(checkpoint=Path(args.checkpoint).name, device=device, max_seqs=args.max_seqs,
                block_size=args.block_size, num_blocks=sched.allocator.num_blocks,
                temperature=args.temperature, max_new=args.max_new,
                history_turns=args.history_turns)
    budget = cfg.block_size - args.max_new
    server = ThreadingHTTPServer((args.host, args.port), make_handler(engine, enc, budget, info, args.history_turns))
    log.info('ready: http://%s:%d  (log level %s%s)', args.host, args.port, args.log_level,
             f', also -> {args.log_file}' if args.log_file else '')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
