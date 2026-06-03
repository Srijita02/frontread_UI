"""Provider-agnostic model client used by the Streamlit UI and CLI."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any


@dataclass
class ModelSpec:
    provider: str
    model: str
    api_key: str | None = None
    display_name: str | None = None

    @property
    def label(self) -> str:
        return self.display_name or f"{self.provider}:{self.model}"


def call_model(
    spec: ModelSpec | dict[str, Any],
    system: str,
    user: str,
    max_tokens: int = 3000,
    temperature: float = 0.3,
) -> str:
    """Dispatch a chat-style model call to OpenAI, Anthropic, Google, Ollama, or demo."""
    if isinstance(spec, dict):
        spec = ModelSpec(**spec)
    provider = spec.provider.lower().strip()
    if provider == "openai":
        return _call_openai(spec, system, user, max_tokens, temperature)
    if provider == "anthropic":
        return _call_anthropic(spec, system, user, max_tokens, temperature)
    if provider == "google":
        return _call_google(spec, system, user, max_tokens, temperature)
    if provider == "ollama":
        return _call_ollama(spec, system, user, max_tokens, temperature)
    if provider == "demo":
        return _call_demo(system, user)
    raise ValueError(f"Unknown provider: {provider}")


def _required_key(spec: ModelSpec, env_name: str) -> str:
    key = spec.api_key or os.getenv(env_name, "")
    if not key:
        raise RuntimeError(f"Missing API key. Paste a key in the UI or set {env_name}.")
    return key


def _call_openai(spec: ModelSpec, system: str, user: str, max_tokens: int, temperature: float) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=_required_key(spec, "OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model=spec.model,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return response.choices[0].message.content or ""


def _call_anthropic(spec: ModelSpec, system: str, user: str, max_tokens: int, temperature: float) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=_required_key(spec, "ANTHROPIC_API_KEY"))
    response = client.messages.create(
        model=spec.model,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return "".join(block.text for block in response.content if getattr(block, "type", "text") == "text")


def _call_google(spec: ModelSpec, system: str, user: str, max_tokens: int, temperature: float) -> str:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=_required_key(spec, "GOOGLE_API_KEY"))
    response = client.models.generate_content(
        model=spec.model,
        contents=user,
        config=types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=max_tokens,
            temperature=temperature,
        ),
    )
    return response.text or ""


def _call_ollama(spec: ModelSpec, system: str, user: str, max_tokens: int, temperature: float) -> str:
    import requests

    payload = {
        "model": spec.model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "options": {"num_predict": max_tokens, "temperature": temperature},
    }
    response = requests.post("http://localhost:11434/api/chat", json=payload, timeout=600)
    response.raise_for_status()
    data = response.json()
    return data.get("message", {}).get("content", "")


def _call_demo(system: str, user: str) -> str:
    """Small deterministic fake backend so evaluators can test the UI without API keys."""
    text = f"{system}\n\n{user}".lower()
    if "return json" in text or "respond only with json" in text:
        dims = re.findall(r'"([a-z_]+)"\s*:', user)
        if not dims:
            dims = re.findall(r"-\s*([a-z_]+):", user)
        if not dims:
            dims = [
                "age_appropriateness",
                "topic_relevance",
                "engagement",
                "paragraph_consistency",
                "new_concepts_control",
                "british_english",
                "narrative_arc",
                "vocabulary_fit",
            ]
        payload = {dim: {"score": 4, "justification": "Demo score only; replace with a real evaluator model."} for dim in dims}
        return json.dumps(payload, indent=2)

    if "literal comprehension questions" in text or "now generate" in text and "questions" in text:
        q_count_match = re.search(r"generate\s+(\d+)\s+new literal comprehension questions", user, flags=re.IGNORECASE)
        q_count = int(q_count_match.group(1)) if q_count_match else 4
        blocks = []
        for i in range(1, q_count + 1):
            correct = "ABCD"[(i - 1) % 4]
            blocks.append(
                f"""{i}. What detail is stated in the passage for demo question {i}?
A: Demo option A
B: Demo option B
C: Demo option C
D: Demo option D
Correct: {correct}
Text reference: Demo reference sentence from the passage."""
            )
        return "\n\n".join(blocks)

    grade = _extract_value(user, "Grade") or "selected grade"
    topic = _extract_value(user, "Topic") or "the selected topic"
    target = int(_extract_value(user, "Target word count", numeric=True) or 180)
    seed = (
        f"This demo passage is about {topic}. It is written for {grade}. "
        "A real model will replace this text with a richer FrontRead passage. "
        "The paragraph uses British English and keeps the idea clear. "
    )
    words = seed.split()
    while len(words) < target:
        words.extend(seed.split())
    body = " ".join(words[:target])
    return f"Title:\nDemo passage about {topic}\n\nBody text:\n[Word count: {target}]\n{body}"


def _extract_value(prompt: str, label: str, numeric: bool = False) -> str | None:
    if numeric:
        match = re.search(rf"{re.escape(label)}[^\n:]*:\s*(?:exactly\s*)?(\d+)", prompt, flags=re.IGNORECASE)
    else:
        match = re.search(rf"{re.escape(label)}\s*:\s*([^\n]+)", prompt, flags=re.IGNORECASE)
    return match.group(1).strip() if match else None
