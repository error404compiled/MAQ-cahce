import sys
import uuid
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

from maqcache import cache


def _install_fake_openai():
    fake = ModuleType("openai")
    fake.OpenAI = SimpleNamespace
    fake.AsyncOpenAI = SimpleNamespace
    sys.modules["openai"] = fake


_install_fake_openai()

with patch("maqcache.utils._check_library", return_value=True):
    from maqcache.adapter import maq_slm
    from maqcache.adapter.openai_client import (
        get_message_from_openai_client_answer,
        get_stream_message_from_openai_client_answer,
    )


def _make_chat_completion(text, model="qwen-3.8-27b"):
    message = SimpleNamespace(role="assistant", content=text)
    choice = SimpleNamespace(index=0, finish_reason="stop", message=message)
    return SimpleNamespace(
        id="chatcmpl_test",
        choices=[choice],
        created=0,
        model=model,
        object="chat.completion",
    )


def test_normal_maq_slm():
    cache.init()
    question = "calculate 1+3 " + uuid.uuid4().hex
    expect_answer = "the result is 4"

    fake_chat = SimpleNamespace(
        completions=SimpleNamespace(
            create=Mock(return_value=_make_chat_completion(expect_answer))
        )
    )
    fake_client = SimpleNamespace(chat=fake_chat)
    maq_slm.ChatCompletion.client = fake_client

    response = maq_slm.ChatCompletion.create(
        model="qwen-3.8-27b",
        messages=[{"role": "user", "content": question}],
    )
    assert get_message_from_openai_client_answer(response) == expect_answer

    response = maq_slm.ChatCompletion.create(
        model="qwen-3.8-27b",
        messages=[{"role": "user", "content": question}],
    )
    assert get_message_from_openai_client_answer(response) == expect_answer
    fake_chat.completions.create.assert_called_once()


@pytest.mark.asyncio
async def test_normal_maq_slm_async():
    cache.init()
    question = "calculate 1+3 " + uuid.uuid4().hex
    expect_answer = "the result is 4"

    fake_chat = SimpleNamespace(
        completions=SimpleNamespace(
            create=AsyncMock(return_value=_make_chat_completion(expect_answer))
        )
    )
    fake_client = SimpleNamespace(chat=fake_chat)
    maq_slm.ChatCompletion.aclient = fake_client

    response = await maq_slm.ChatCompletion.acreate(
        model="qwen-3.8-27b",
        messages=[{"role": "user", "content": question}],
    )
    assert get_message_from_openai_client_answer(response) == expect_answer

    response = await maq_slm.ChatCompletion.acreate(
        model="qwen-3.8-27b",
        messages=[{"role": "user", "content": question}],
    )
    assert get_message_from_openai_client_answer(response) == expect_answer
    fake_chat.completions.create.assert_awaited_once()


def test_stream_maq_slm():
    cache.init()
    question = "calculate 1+1 " + uuid.uuid4().hex
    expect_answer = "the result is 2"

    def _stream():
        yield SimpleNamespace(
            choices=[
                SimpleNamespace(
                    index=0,
                    finish_reason=None,
                    delta=SimpleNamespace(role="assistant", content=expect_answer),
                )
            ]
        )

    fake_chat = SimpleNamespace(completions=SimpleNamespace(create=Mock(return_value=_stream())))
    fake_client = SimpleNamespace(chat=fake_chat)
    maq_slm.ChatCompletion.client = fake_client

    response = maq_slm.ChatCompletion.create(
        model="qwen-3.8-27b",
        stream=True,
        messages=[{"role": "user", "content": question}],
    )
    assert "".join(get_stream_message_from_openai_client_answer(c) for c in response) == expect_answer

    response = maq_slm.ChatCompletion.create(
        model="qwen-3.8-27b",
        stream=True,
        messages=[{"role": "user", "content": question}],
    )
    assert "".join(get_stream_message_from_openai_client_answer(c) for c in response) == expect_answer
    fake_chat.completions.create.assert_called_once()


def test_maq_slm_client_defaults_to_gateway(monkeypatch):
    monkeypatch.delenv("MAQ_SLM_BASE_URL", raising=False)
    monkeypatch.setenv("MAQ_SLM_API_KEY", "sk-test-key")
    maq_slm.ChatCompletion.client = None

    client = maq_slm.ChatCompletion._sync_client()
    assert client.base_url == maq_slm.DEFAULT_BASE_URL
    assert client.api_key == "sk-test-key"
