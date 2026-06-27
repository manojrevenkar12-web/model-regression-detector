import os
from typing import Any

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

_client = AsyncOpenAI(
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
)


async def call_llm(
    messages: list[dict[str, str]],
    model: str,
    temperature: float = 0.0,
    **kwargs: Any,
) -> str:
    response = await _client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        **kwargs,
    )
    return response.choices[0].message.content or ""
