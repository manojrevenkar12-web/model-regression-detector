import json
import re

from src.llm import call_llm
from src.models import ClassifierOutput, PromptConfig


async def classify_email(text: str, prompt_config: PromptConfig) -> ClassifierOutput:
    messages = _build_messages(text, prompt_config)
    raw = await call_llm(messages, model=prompt_config.model, temperature=0.0)
    return _parse_output(raw)


def _build_messages(text: str, prompt_config: PromptConfig) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = [
        {"role": "system", "content": prompt_config.system_prompt}
    ]
    for ex in prompt_config.few_shot_examples:
        messages.append({"role": "user", "content": ex.email})
        messages.append({
            "role": "assistant",
            "content": json.dumps({"category": ex.category, "summary": ex.summary}),
        })
    messages.append({"role": "user", "content": text})
    return messages


def _parse_output(raw: str) -> ClassifierOutput:
    # Strip markdown code fences if the model wraps its JSON response
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.DOTALL)
    return ClassifierOutput.model_validate_json(cleaned)
