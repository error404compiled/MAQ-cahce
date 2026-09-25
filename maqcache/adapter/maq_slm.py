import os

from maqcache.adapter import openai_client
from maqcache.utils import import_openai_client

import_openai_client()

# pylint: disable=C0413
from openai import AsyncOpenAI, OpenAI

DEFAULT_BASE_URL = "https://indiaai.maqsoftware.net/v1"


class ChatCompletion(openai_client.ChatCompletion):
    """Chat completion wrapper for MAQ SLM, MAQ Softwares' internal
    OpenAI-compatible LLM gateway.

    The gateway speaks the standard OpenAI Chat Completions API, so this
    just points the modern OpenAI SDK client at MAQ's endpoint instead of
    api.openai.com. Configure it with:

    - ``MAQ_SLM_API_KEY``: your gateway API key (required).
    - ``MAQ_SLM_BASE_URL``: override the gateway URL (defaults to
      ``https://indiaai.maqsoftware.net/v1``).

    Example:
        .. code-block:: python

            from maqcache import cache
            from maqcache.adapter import maq_slm

            cache.init()

            response = maq_slm.ChatCompletion.create(
                model="qwen-3.8-27b",
                messages=[{"role": "user", "content": "what's github"}],
            )
            answer = response.choices[0].message.content
    """

    @classmethod
    def _sync_client(cls):
        if cls.client is not None:
            return cls.client
        return OpenAI(
            api_key=os.getenv("MAQ_SLM_API_KEY") or os.getenv("MAQ_API_KEY"),
            base_url=os.getenv("MAQ_SLM_BASE_URL", DEFAULT_BASE_URL),
        )

    @classmethod
    def _async_client(cls):
        if cls.aclient is not None:
            return cls.aclient
        return AsyncOpenAI(
            api_key=os.getenv("MAQ_SLM_API_KEY") or os.getenv("MAQ_API_KEY"),
            base_url=os.getenv("MAQ_SLM_BASE_URL", DEFAULT_BASE_URL),
        )
