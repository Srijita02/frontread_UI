"""AI-based qualitative evaluation prompts and helpers."""

from __future__ import annotations

from typing import Any

from .config import QUESTION_EVAL_DIMS, TEXT_EVAL_DIMS
from .metrics import ai_overall_score
from .model_client import ModelSpec, call_model
from .parsers import extract_json_object, normalise_eval_payload
from .prompt_templates import (
    EVALUATION_SYSTEM_TEMPLATE,
    build_question_evaluation_prompts,
    build_text_evaluation_prompts,
)

# Backwards-compatible constant used by earlier code/tests.
EVAL_SYSTEM = EVALUATION_SYSTEM_TEMPLATE


def build_text_eval_prompt(
    title: str,
    body: str,
    params: dict[str, Any],
    metrics: dict[str, Any],
    prompt_overrides: dict[str, str] | None = None,
) -> str:
    """Return the rendered user prompt for text evaluation."""
    _system, user_prompt = build_text_evaluation_prompts(
        title=title,
        body=body,
        params=params,
        metrics=metrics,
        prompt_overrides=prompt_overrides,
    )
    return user_prompt


def build_question_eval_prompt(
    title: str,
    body: str,
    questions: list[dict[str, Any]],
    params: dict[str, Any],
    q_metrics: dict[str, Any],
    prompt_overrides: dict[str, str] | None = None,
) -> str:
    """Return the rendered user prompt for question evaluation."""
    _system, user_prompt = build_question_evaluation_prompts(
        title=title,
        body=body,
        questions=questions,
        params=params,
        q_metrics=q_metrics,
        prompt_overrides=prompt_overrides,
    )
    return user_prompt


def evaluate_text_ai(
    evaluator: ModelSpec,
    title: str,
    body: str,
    params: dict[str, Any],
    metrics: dict[str, Any],
    temperature: float = 0.0,
    prompt_overrides: dict[str, str] | None = None,
) -> dict[str, Any]:
    system_prompt, user_prompt = build_text_evaluation_prompts(
        title=title,
        body=body,
        params=params,
        metrics=metrics,
        prompt_overrides=prompt_overrides,
    )
    raw = call_model(evaluator, system_prompt, user_prompt, max_tokens=2500, temperature=temperature)
    try:
        payload = extract_json_object(raw)
        out = normalise_eval_payload(payload, TEXT_EVAL_DIMS)
    except Exception as exc:
        out = {dim: {"score": None, "justification": "Evaluator JSON parsing failed."} for dim in TEXT_EVAL_DIMS}
        out["_error"] = str(exc)
        out["_raw"] = raw
    out["_overall_score"] = ai_overall_score(out)
    return out


def evaluate_questions_ai(
    evaluator: ModelSpec,
    title: str,
    body: str,
    questions: list[dict[str, Any]],
    params: dict[str, Any],
    q_metrics: dict[str, Any],
    temperature: float = 0.0,
    prompt_overrides: dict[str, str] | None = None,
) -> dict[str, Any]:
    system_prompt, user_prompt = build_question_evaluation_prompts(
        title=title,
        body=body,
        questions=questions,
        params=params,
        q_metrics=q_metrics,
        prompt_overrides=prompt_overrides,
    )
    raw = call_model(evaluator, system_prompt, user_prompt, max_tokens=3000, temperature=temperature)
    try:
        payload = extract_json_object(raw)
        out = normalise_eval_payload(payload, QUESTION_EVAL_DIMS)
    except Exception as exc:
        out = {dim: {"score": None, "justification": "Evaluator JSON parsing failed."} for dim in QUESTION_EVAL_DIMS}
        out["_error"] = str(exc)
        out["_raw"] = raw
    out["_overall_score"] = ai_overall_score(out)
    return out
