import os
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

_client = AsyncOpenAI(
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
)


@dataclass
class LLMResponse:
    content: str
    prompt_tokens: int
    completion_tokens: int


async def call_llm_full(
    messages: list[dict[str, str]],
    model: str,
    temperature: float = 0.0,
    **kwargs: Any,
) -> LLMResponse:
    response = await _client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        **kwargs,
    )
    usage = response.usage
    return LLMResponse(
        content=response.choices[0].message.content or "",
        prompt_tokens=usage.prompt_tokens if usage else 0,
        completion_tokens=usage.completion_tokens if usage else 0,
    )


async def call_llm(
    messages: list[dict[str, str]],
    model: str,
    temperature: float = 0.0,
    **kwargs: Any,
) -> str:
    return (await call_llm_full(messages, model, temperature, **kwargs)).content
