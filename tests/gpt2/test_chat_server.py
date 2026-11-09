"""The FastAPI app: the chat page's API, the OpenAI-compatible routes, and the Engine
thread that is the scheduler's single writer."""

import threading

import pytest
import torch

pytest.importorskip('fastapi')
from fastapi.testclient import TestClient

from app import LLM, SamplingParams
from app.chat_server import CHAT_STOPS, create_app, cut, render
from model.gpt2 import GPT, GPTConfig

TINY = GPTConfig(vocab_size=50304, n_layer=2, n_head=2, n_embd=32, block_size=128)
GREEDY = SamplingParams(max_new_tokens=6)


@pytest.fixture(scope='module')
def llm(tmp_path_factory):
    torch.manual_seed(0)
    path = tmp_path_factory.mktemp('ck') / 'tiny.pt'
    torch.save({'config': vars(TINY), 'model': GPT(TINY).state_dict()}, path)
    return LLM(str(path), engine='paged', device='cpu', max_seqs=4, block_size=4)


@pytest.fixture
def client(llm):
    with TestClient(create_app(llm, GREEDY, info={'checkpoint': 'tiny.pt'})) as c:   # runs lifespan
        yield c


def test_chat_page_and_info(client):
    assert '<title>GPT-2 Chat</title>' in client.get('/').text
    info = client.get('/api/info').json()
    assert info['engine'] == 'paged' and info['checkpoint'] == 'tiny.pt'
    assert info['max_seqs'] == 4, 'the batch limit must follow the engine, not a fallback'


def test_ui_chat_matches_a_direct_generate(client, llm):
    r = client.post('/api/chat', json={'messages': [{'role': 'user', 'content': 'Hello'}]})
    assert r.status_code == 200
    prompt = render([{'role': 'user', 'content': 'Hello'}], llm.enc, TINY.block_size - 6)
    expected = cut(llm.generate(prompt, GREEDY), CHAT_STOPS)[0].strip() or '…'
    assert r.json()['reply'] == expected
    assert r.json()['stats']['batch_size'] >= 1


def test_ui_chat_rejects_an_empty_conversation(client):
    assert client.post('/api/chat', json={'messages': []}).status_code == 422


def test_openai_completions(client, llm):
    r = client.post('/v1/completions', json={'prompt': ['Hello', 'A'], 'max_tokens': 5,
                                             'temperature': 0})
    body = r.json()
    assert r.status_code == 200 and body['object'] == 'text_completion'
    assert [c['text'] for c in body['choices']] == llm.generate(['Hello', 'A'],
                                                                SamplingParams(max_new_tokens=5))
    assert body['usage']['completion_tokens'] <= 10
    assert all(c['finish_reason'] in ('stop', 'length') for c in body['choices'])


def test_openai_completions_honours_stop(client):
    r = client.post('/v1/completions', json={'prompt': 'Hello', 'max_tokens': 8, 'temperature': 0,
                                             'stop': ['e', 'a', 'o', ' ']})
    choice = r.json()['choices'][0]
    assert not any(s in choice['text'] for s in 'eao ') and choice['finish_reason'] in ('stop', 'length')


def test_openai_chat_completions(client):
    r = client.post('/v1/chat/completions', json={
        'messages': [{'role': 'user', 'content': 'Hi there'}], 'max_tokens': 6, 'temperature': 0})
    body = r.json()
    assert r.status_code == 200 and body['object'] == 'chat.completion'
    msg = body['choices'][0]['message']
    assert msg['role'] == 'assistant' and '\n' not in msg['content']


def test_prompt_too_long_is_a_400(client):
    r = client.post('/v1/completions', json={'prompt': 'word ' * 200, 'max_tokens': 8})
    assert r.status_code == 400


def test_only_the_engine_thread_touches_the_scheduler(client, llm, monkeypatch):
    """Requests arrive on server threads; generate() must always run on 'engine'."""
    seen, original = set(), llm.engine.generate

    def spy(*a, **k):
        seen.add(threading.current_thread().name)
        return original(*a, **k)

    monkeypatch.setattr(llm.engine, 'generate', spy)
    threads = [threading.Thread(target=client.post, args=('/v1/completions',),
                                kwargs={'json': {'prompt': f'Hi {i}', 'max_tokens': 3, 'temperature': 0}})
               for i in range(4)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert seen == {'engine'}


def test_engine_groups_a_batch_by_sampling_params(llm):
    """Different settings in one gathered batch run as separate generate() calls."""
    from app.chat_server import Engine
    calls = []
    original = llm.engine.generate

    def spy(prompts, sampling_params):
        calls.append((len(prompts), sampling_params))
        return original(prompts, sampling_params)

    llm.engine.generate = spy
    try:
        e = Engine(llm, max_batch=8, gather_window=0.2)
        greedy, warm = SamplingParams(max_new_tokens=3), SamplingParams(max_new_tokens=3, temperature=0.9)
        futures = [e.submit('a', greedy), e.submit('b', warm), e.submit('c', greedy)]
        e.start()
        [f.result(timeout=30) for f in futures]
    finally:
        del llm.engine.generate
    assert sorted(calls, key=lambda c: c[0]) == [(1, warm), (2, greedy)]
