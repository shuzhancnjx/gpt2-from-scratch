"""A chat UI and an OpenAI-compatible API over one LLM, served by FastAPI.

    from app import LLM
    from app.chat_server import create_app
    app = create_app(LLM('runs/keep/gpt2_124M_step013147.pt'))      # then: uvicorn

Routes:

    GET  /                      the chat page (chat.html)
    GET  /api/info              engine and settings, for the page header
    POST /api/chat              {"messages": [...]}  -> {"reply", "stats"}   (the page uses this)
    POST /v1/completions        OpenAI-style text completion
    POST /v1/chat/completions   OpenAI-style chat completion
    GET  /docs                  interactive API docs (FastAPI)

One writer. FastAPI serves every request concurrently (async), but only ONE thread --
the RequestBatcher -- ever touches the inference engine, its block pool and its CUDA
graphs (request-lifecycle.md §20). An endpoint hands a job to the batcher through a queue
and awaits a future; the batcher gathers jobs that arrive close together, groups them by
SamplingParams (a batch shares one sampler -- and one captured graph), and runs one
generate() per group.

Batching is between runs: a job arriving while a run is going waits for the next run.
Admitting into a running batch needs a step-level engine API (a later milestone).

The model is a base GPT-2, not a chat model: chat messages are rendered as a
"User: / Assistant:" transcript and the reply is cut at the end of its first line.
"""

import asyncio
import logging
import queue
import threading
import time
import uuid
from concurrent.futures import Future
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from inference.sampling import SamplingParams

log = logging.getLogger('chat_server')

PAGE = (Path(__file__).parent / 'chat.html').read_text()

# Plain transcript, no few-shot examples: tried on this 124M base model, examples were
# copied into answers rather than imitated as a style. Question answering needs SFT
# (learning plan, Phase 4).
PREAMBLE = 'Below is a conversation between a curious user and a knowledgeable assistant.\n\n'
# A base model rambles on past its answer; one line reads as one chat turn.
CHAT_STOPS = ('\n',)


# -- prompt rendering ----------------------------------------------------------------

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


def cut(text, stops):
    """Text up to the first stop string; True if a stop was found."""
    hits = [i for i in (text.find(s) for s in stops) if i != -1]
    return (text[:min(hits)], True) if hits else (text, False)


# -- the RequestBatcher thread: the single writer ----------------------------------------

class RequestBatcher(threading.Thread):
    """Gathers requests into batches and runs them on the inference engine (llm.engine),
    from this one thread only. Not an engine itself: it decides WHICH requests run
    together; the engine decides how a batch runs."""

    def __init__(self, llm, max_batch, gather_window=0.03):
        super().__init__(daemon=True, name='batcher')
        self.llm = llm
        self.max_batch, self.gather_window = max_batch, gather_window
        self.inbox = queue.Queue()
        self.runs = 0

    def submit(self, prompt, sampling_params):
        """Thread-safe: queue a job, return a Future the caller can await."""
        job = dict(prompt=prompt, sampling_params=sampling_params, future=Future())
        self.inbox.put(job)
        return job['future']

    def run(self):
        while True:
            jobs = [self.inbox.get()]                    # blocks when idle: no spinning
            time.sleep(self.gather_window)               # let near-simultaneous requests join
            while len(jobs) < self.max_batch:
                try:
                    jobs.append(self.inbox.get_nowait())
                except queue.Empty:
                    break
            groups = {}                                  # one generate() per sampler setting
            for job in jobs:
                groups.setdefault(job['sampling_params'], []).append(job)
            for sampling_params, group in groups.items():
                self._run_group(sampling_params, group, len(jobs))

    def _run_group(self, sampling_params, group, gathered):
        engine = self.llm.engine
        log.info('batch %d: %d request(s) -> engine (%s)', self.runs + 1, len(group), sampling_params)
        t0 = time.perf_counter()
        try:
            outs = engine.generate([j['prompt'] for j in group], sampling_params)
        except Exception as e:                           # report to every caller, keep serving
            for job in group:
                job['future'].set_exception(e)
            return
        seconds = time.perf_counter() - t0
        self.runs += 1
        generated = getattr(engine, 'generated', {})
        stats = getattr(engine, 'stats', None)
        log.info('batch %d done in %.2fs: %d decode steps, peak %s blocks / %s requests resident',
                 self.runs, seconds, getattr(stats, 'decode_steps', 0),
                 getattr(stats, 'peak_blocks', '-'), getattr(stats, 'peak_active', '-'))
        for i, (job, text) in enumerate(zip(group, outs)):
            # schedulers record ids per prompt; the static engine doesn't, so re-encode
            tokens = len(generated[i]) if i in generated else len(self.llm.enc.encode(text))
            job['future'].set_result(dict(
                text=text, tokens=tokens,
                stats=dict(batch_size=len(group), gathered=gathered, seconds=round(seconds, 2),
                           tokens=tokens,
                           resident=getattr(stats, 'peak_active', None),
                           blocks=getattr(stats, 'peak_blocks', None))))


# -- request and response shapes --------------------------------------------------------------

class Message(BaseModel):
    role: str
    content: str


class UIChatRequest(BaseModel):
    messages: list[Message] = Field(min_length=1)


class _Sampling(BaseModel):
    model: str | None = None
    temperature: float = 1.0
    top_p: float | None = None
    top_k: int | None = None                         # not in OpenAI's API; vLLM accepts it too
    stop: str | list[str] | None = None


class CompletionRequest(_Sampling):
    prompt: str | list[str]
    max_tokens: int = Field(16, ge=1)


class ChatCompletionRequest(_Sampling):
    messages: list[Message] = Field(min_length=1)
    max_tokens: int = Field(64, ge=1)


def _stops(stop):
    return () if stop is None else (stop,) if isinstance(stop, str) else tuple(stop)


def _clip(text, n=100):
    text = ' '.join(str(text).split())
    return text if len(text) <= n else text[:n] + '…'


# -- the app ------------------------------------------------------------------------------

def create_app(llm, sampling_params=None, history_turns=0, gather_window=0.03, info=None):
    """sampling_params: the SamplingParams the chat page uses (default: temperature 0.8, top-k 40,
    64 new tokens). The /v1 routes take theirs from each request."""
    sampling_params = sampling_params or SamplingParams(max_new_tokens=64, temperature=0.8, top_k=40)
    max_batch = getattr(llm.engine, 'max_seqs', 8)           # the static engine has none
    batcher = RequestBatcher(llm, max_batch=max_batch, gather_window=gather_window)
    enc, context = llm.enc, llm.config.block_size
    name = (info or {}).get('checkpoint', 'gpt2')

    @asynccontextmanager
    async def lifespan(app):
        batcher.start()
        log.info('batcher thread started (%s, up to %d requests per batch)', llm, max_batch)
        yield

    app = FastAPI(title='GPT-2 from scratch', lifespan=lifespan)
    app.state.batcher = batcher

    async def generate(prompt, p):
        if len(enc.encode(prompt)) + p.max_new_tokens > context:
            raise HTTPException(400, f'prompt + max_tokens exceeds the {context}-token context')
        try:
            return await asyncio.wrap_future(batcher.submit(prompt, p))
        except ValueError as e:
            raise HTTPException(400, str(e))
        except Exception as e:
            log.error('request failed: %s', e)
            raise HTTPException(500, f'{type(e).__name__}: {e}')

    def settings(req):
        return SamplingParams(max_new_tokens=req.max_tokens, temperature=req.temperature,
                              top_k=req.top_k, top_p=req.top_p)

    def usage(prompt_tokens, completion_tokens):
        return dict(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens)

    # -- the chat page ----------------------------------------------------------------------

    @app.get('/', response_class=HTMLResponse)
    async def page():
        return PAGE

    @app.get('/api/info')
    async def api_info():
        return dict(info or {}, device=llm.device, engine=llm.engine_name,
                    max_seqs=max_batch, temperature=sampling_params.temperature,
                    max_new=sampling_params.max_new_tokens, history_turns=history_turns, runs=batcher.runs)

    @app.post('/api/chat')
    async def api_chat(req: UIChatRequest):
        messages = [m.model_dump() for m in req.messages]
        prompt = render(messages, enc, context - sampling_params.max_new_tokens, history_turns)
        log.info('user: %s', _clip(messages[-1]['content']))
        log.debug('prompt (%d tokens):\n%s', len(enc.encode(prompt)), prompt)
        out = await generate(prompt, sampling_params)
        reply = cut(out['text'], CHAT_STOPS)[0].strip() or '…'
        log.debug('raw output (before the cut at the first newline):\n%s', out['text'])
        log.info('reply: %s  [%s]', _clip(reply), ', '.join(f'{k}={v}' for k, v in out['stats'].items()))
        return dict(reply=reply, stats=out['stats'])

    # -- OpenAI-compatible --------------------------------------------------------------------

    @app.post('/v1/completions')
    async def completions(req: CompletionRequest):
        prompts = [req.prompt] if isinstance(req.prompt, str) else req.prompt
        p = settings(req)
        outs = await asyncio.gather(*(generate(x, p) for x in prompts))   # batched by the RequestBatcher
        choices, n_prompt, n_gen = [], 0, 0
        for i, (x, out) in enumerate(zip(prompts, outs)):
            text, stopped = cut(out['text'], _stops(req.stop))
            done = stopped or out['tokens'] < p.max_new_tokens       # ended early: <|endoftext|>
            choices.append(dict(index=i, text=text, logprobs=None,
                                finish_reason='stop' if done else 'length'))
            n_prompt += len(enc.encode(x))
            n_gen += out['tokens']
        return dict(id=f'cmpl-{uuid.uuid4().hex[:24]}', object='text_completion',
                    created=int(time.time()), model=req.model or name,
                    choices=choices, usage=usage(n_prompt, n_gen))

    @app.post('/v1/chat/completions')
    async def chat_completions(req: ChatCompletionRequest):
        p = settings(req)
        messages = [m.model_dump() for m in req.messages]
        prompt = render(messages, enc, context - p.max_new_tokens, history_turns=None)
        out = await generate(prompt, p)
        text, stopped = cut(out['text'], CHAT_STOPS + _stops(req.stop))
        done = stopped or out['tokens'] < p.max_new_tokens
        return dict(id=f'chatcmpl-{uuid.uuid4().hex[:24]}', object='chat.completion',
                    created=int(time.time()), model=req.model or name,
                    choices=[dict(index=0, message=dict(role='assistant', content=text.strip()),
                                  finish_reason='stop' if done else 'length')],
                    usage=usage(len(enc.encode(prompt)), out['tokens']))

    return app
