"""Decoupled FrontRead generation and evaluation pipeline for the Streamlit UI."""

from __future__ import annotations

import random
import re
import time
import uuid
from copy import deepcopy
from datetime import datetime
from typing import Any, Callable

from .config import (
    DEFAULT_APPROVAL_THRESHOLD,
    DEFAULT_MAX_REVISIONS,
    DEFAULT_MIN_DIMENSION_SCORE,
    DEFAULT_WORD_TOLERANCE,
    GRADE_LEVELS,
    TEXT_TYPES,
)
from .evaluation import evaluate_questions_ai, evaluate_text_ai
from .metrics import (
    approval_flags,
    describe_text_metric_problems,
    question_metrics,
    strip_word_count_tag,
    text_metrics,
)
from .model_client import ModelSpec, call_model
from .parsers import parse_questions_output, parse_text_output
from .prompt_templates import (
    TEXT_REVISION_PROMPT,
    build_question_generation_prompts,
    build_text_generation_prompts,
)

ProgressCallback = Callable[[str, str, int, int], None]


TEXT_EVAL_NOT_RUN = {"_overall_score": None, "_skipped": True, "_status": "not_run"}
QUESTION_EVAL_NOT_RUN = {"_overall_score": None, "_skipped": True, "_status": "not_run"}


def public_model_spec(spec: ModelSpec | None) -> dict[str, Any]:
    if spec is None:
        return {"provider": "", "model": "", "display_name": ""}
    return {
        "provider": spec.provider,
        "model": spec.model,
        "display_name": spec.display_name or f"{spec.provider}:{spec.model}",
    }


def safe_slug(text: str, max_len: int = 60) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip().lower()).strip("_")
    return (slug or "run")[:max_len]


def build_params(
    grade_key: str,
    topic: str,
    word_count: int,
    question_count: int,
    text_type_key: str = "mixed",
    text_format: str | None = None,
) -> dict[str, Any]:
    grade = GRADE_LEVELS[grade_key]
    text_type = TEXT_TYPES[text_type_key]
    chosen_format = text_format or random.choice(text_type["formats"])
    return {
        "grade_key": grade_key,
        "grade_label": grade["label"],
        "lix_target": grade["lix_target"],
        "lix_min": grade["lix_range"][0],
        "lix_max": grade["lix_range"][1],
        "topic": topic,
        "subtopic": topic,
        "text_type_key": text_type_key,
        "text_type_label": text_type["label"],
        "text_type": text_type["label"],
        "text_format": chosen_format,
        "word_count": int(word_count),
        "question_count": int(question_count),
        "sentence_length_avg": grade["sentence_length_avg"],
        "vocabulary_notes": grade["vocabulary_notes"],
    }


def _blank_approval() -> dict[str, Any]:
    return {
        "text_metric_ok": False,
        "text_ai_ok": False,
        "text_approved": False,
        "question_metric_ok": False,
        "question_ai_ok": False,
        "questions_approved": False,
        "overall_approved": False,
        "text_ai_overall_score": None,
        "text_ai_min_score": None,
        "question_ai_overall_score": None,
        "question_ai_min_score": None,
    }


def refresh_approval(
    result: dict[str, Any],
    approval_threshold: float = DEFAULT_APPROVAL_THRESHOLD,
    min_dimension_score: float = DEFAULT_MIN_DIMENSION_SCORE,
) -> dict[str, Any]:
    """Recompute approval flags from whatever stages have already run."""
    text_metrics_row = result.get("text", {}).get("metrics", {})
    question_metrics_row = result.get("questions", {}).get("metrics", {})
    if not text_metrics_row and not question_metrics_row:
        result["approval"] = _blank_approval()
        return result["approval"]

    result["approval"] = approval_flags(
        text_metrics_row,
        question_metrics_row,
        result.get("text_ai_eval", TEXT_EVAL_NOT_RUN),
        result.get("question_ai_eval", QUESTION_EVAL_NOT_RUN),
        approval_threshold=approval_threshold,
        min_dimension_score=min_dimension_score,
    )
    return result["approval"]


def _base_result(
    topic: str,
    grade_key: str,
    params: dict[str, Any],
    text_generator: ModelSpec | None = None,
    question_generator: ModelSpec | None = None,
    text_evaluator: ModelSpec | None = None,
    question_evaluator: ModelSpec | None = None,
) -> dict[str, Any]:
    models = {
        "text_generator": public_model_spec(text_generator),
        "question_generator": public_model_spec(question_generator),
        "text_evaluator": public_model_spec(text_evaluator),
        "question_evaluator": public_model_spec(question_evaluator),
    }
    return {
        "run_item_id": str(uuid.uuid4()),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "topic": topic,
        "grade_key": grade_key,
        "grade_label": params.get("grade_label", GRADE_LEVELS.get(grade_key, {}).get("label", grade_key)),
        "models": models,
        # Backwards-compatible aliases used by older exports/UI code.
        "generator": models["text_generator"],
        "evaluator": models["text_evaluator"],
        "params": params,
        "text_ai_eval": deepcopy(TEXT_EVAL_NOT_RUN),
        "question_ai_eval": deepcopy(QUESTION_EVAL_NOT_RUN),
        "approval": _blank_approval(),
        "prompting": {
            "text_generation_strategy": "few_shot",
            "question_generation_strategy": "few_shot",
            "custom_prompts_used": False,
        },
    }


def _set_model(result: dict[str, Any], role: str, spec: ModelSpec) -> None:
    result.setdefault("models", {})[role] = public_model_spec(spec)
    if role == "text_generator":
        result["generator"] = public_model_spec(spec)
    if role == "text_evaluator":
        result["evaluator"] = public_model_spec(spec)


def generate_text(
    generator: ModelSpec,
    params: dict[str, Any],
    word_tolerance: int = DEFAULT_WORD_TOLERANCE,
    max_revisions: int = DEFAULT_MAX_REVISIONS,
    temperature: float = 0.3,
    prompt_strategy: str = "few_shot",
    prompt_overrides: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Generate one text, then iteratively revise for LIX/word-count compliance."""
    start = time.time()
    system_prompt, prompt = build_text_generation_prompts(
        params,
        strategy=prompt_strategy,
        prompt_overrides=prompt_overrides,
    )
    raw = call_model(
        generator,
        system_prompt,
        prompt,
        max_tokens=max(1200, int(params["word_count"]) * 4),
        temperature=temperature,
    )
    title, body = parse_text_output(raw)
    metrics = text_metrics(body, params, word_tolerance=word_tolerance)
    revision_history = [
        {
            "revision": 0,
            "lix_actual": metrics["lix_actual"],
            "actual_word_count": metrics["actual_word_count"],
            "lix_in_band": metrics["lix_in_band"],
            "word_count_ok": metrics["word_count_ok"],
        }
    ]
    raw_outputs = [raw]

    iteration = 0
    while iteration < int(max_revisions):
        if metrics["lix_in_band"] and metrics["word_count_ok"]:
            break
        iteration += 1
        problems = describe_text_metric_problems(metrics)
        revision_prompt = TEXT_REVISION_PROMPT.format(
            title=title,
            body=body,
            actual_words=metrics["actual_word_count"],
            word_count=params["word_count"],
            lix_actual=metrics["lix_actual"],
            lix_min=params["lix_min"],
            lix_max=params["lix_max"],
            avg_sent_len=metrics["avg_sentence_length"],
            long_word_pct=metrics["long_word_pct"],
            problems=problems,
        )
        revised_raw = call_model(
            generator,
            system_prompt,
            revision_prompt,
            max_tokens=max(1200, int(params["word_count"]) * 4),
            temperature=temperature,
        )
        revised_title, revised_body = parse_text_output(revised_raw)
        raw_outputs.append(revised_raw)
        if revised_body and len(revised_body) > 50:
            title = revised_title or title
            body = revised_body
            metrics = text_metrics(body, params, word_tolerance=word_tolerance)
            revision_history.append(
                {
                    "revision": iteration,
                    "lix_actual": metrics["lix_actual"],
                    "actual_word_count": metrics["actual_word_count"],
                    "lix_in_band": metrics["lix_in_band"],
                    "word_count_ok": metrics["word_count_ok"],
                }
            )
        else:
            break

    return {
        "title": title,
        "body": body,
        "metrics": metrics,
        "params": params,
        "prompt_strategy": prompt_strategy,
        "revision_history": revision_history,
        "final_revision": iteration,
        "raw_outputs": raw_outputs,
        "elapsed_sec": round(time.time() - start, 2),
    }


def generate_questions(
    generator: ModelSpec,
    text_record: dict[str, Any],
    temperature: float = 0.2,
    prompt_strategy: str = "few_shot",
    prompt_overrides: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Generate literal comprehension questions from the finalised text."""
    start = time.time()
    params = text_record["params"]
    clean_body = strip_word_count_tag(text_record["body"])
    system_prompt, prompt = build_question_generation_prompts(
        params=params,
        title=text_record["title"],
        body=clean_body,
        strategy=prompt_strategy,
        prompt_overrides=prompt_overrides,
    )
    raw = call_model(
        generator,
        system_prompt,
        prompt,
        max_tokens=max(1600, int(params["question_count"]) * 420),
        temperature=temperature,
    )
    questions = parse_questions_output(raw)
    q_metrics = question_metrics(questions, int(params["question_count"]))
    return {
        "title": text_record["title"],
        "questions": questions,
        "metrics": q_metrics,
        "raw_output": raw,
        "params": params,
        "prompt_strategy": prompt_strategy,
        "elapsed_sec": round(time.time() - start, 2),
    }


def run_text_stage(
    text_generator: ModelSpec,
    topics: list[str],
    grade_keys: list[str],
    word_count: int,
    question_count: int,
    text_type_key: str = "mixed",
    text_format: str | None = None,
    approval_threshold: float = DEFAULT_APPROVAL_THRESHOLD,
    min_dimension_score: float = DEFAULT_MIN_DIMENSION_SCORE,
    word_tolerance: int = DEFAULT_WORD_TOLERANCE,
    max_revisions: int = DEFAULT_MAX_REVISIONS,
    temperature: float = 0.3,
    prompt_strategy: str = "few_shot",
    prompt_overrides: dict[str, str] | None = None,
    progress_callback: ProgressCallback | None = None,
) -> list[dict[str, Any]]:
    """Stage 1 only: generate texts and Python text metrics."""
    topics = [t.strip() for t in topics if t.strip()]
    total = len(topics) * len(grade_keys)
    results: list[dict[str, Any]] = []
    counter = 0
    for topic in topics:
        for grade_key in grade_keys:
            counter += 1
            label = f"{topic} / {GRADE_LEVELS[grade_key]['label']}"
            params = build_params(grade_key, topic, word_count, question_count, text_type_key, text_format)
            result = _base_result(topic, grade_key, params, text_generator=text_generator)
            result["prompting"]["text_generation_strategy"] = prompt_strategy
            result["prompting"]["custom_prompts_used"] = bool(prompt_overrides)
            if progress_callback:
                progress_callback("text_generation", label, counter, total)
            try:
                result["text"] = generate_text(
                    text_generator,
                    params,
                    word_tolerance=word_tolerance,
                    max_revisions=max_revisions,
                    temperature=temperature,
                    prompt_strategy=prompt_strategy,
                    prompt_overrides=prompt_overrides,
                )
            except Exception as exc:
                result["error"] = f"Text generation failed: {exc}"
            refresh_approval(result, approval_threshold, min_dimension_score)
            results.append(result)
    return results


def run_question_stage(
    results: list[dict[str, Any]],
    question_generator: ModelSpec,
    approval_threshold: float = DEFAULT_APPROVAL_THRESHOLD,
    min_dimension_score: float = DEFAULT_MIN_DIMENSION_SCORE,
    temperature: float = 0.2,
    prompt_strategy: str = "few_shot",
    prompt_overrides: dict[str, str] | None = None,
    overwrite: bool = True,
    progress_callback: ProgressCallback | None = None,
) -> list[dict[str, Any]]:
    """Stage 2 only: generate questions for existing text records."""
    updated = deepcopy(results)
    runnable = [r for r in updated if r.get("text") and not r.get("error")]
    total = len(runnable)
    counter = 0
    for result in updated:
        if not result.get("text") or result.get("error"):
            continue
        if result.get("questions") and not overwrite:
            continue
        counter += 1
        label = f"{result.get('topic')} / {result.get('grade_label')}"
        if progress_callback:
            progress_callback("question_generation", label, counter, total)
        try:
            _set_model(result, "question_generator", question_generator)
            result.setdefault("prompting", {})["question_generation_strategy"] = prompt_strategy
            result.setdefault("prompting", {})["custom_prompts_used"] = bool(prompt_overrides) or result.get("prompting", {}).get("custom_prompts_used", False)
            result["questions"] = generate_questions(
                question_generator,
                result["text"],
                temperature=temperature,
                prompt_strategy=prompt_strategy,
                prompt_overrides=prompt_overrides,
            )
        except Exception as exc:
            result["question_error"] = f"Question generation failed: {exc}"
        refresh_approval(result, approval_threshold, min_dimension_score)
    return updated


def run_text_eval_stage(
    results: list[dict[str, Any]],
    text_evaluator: ModelSpec,
    approval_threshold: float = DEFAULT_APPROVAL_THRESHOLD,
    min_dimension_score: float = DEFAULT_MIN_DIMENSION_SCORE,
    prompt_overrides: dict[str, str] | None = None,
    overwrite: bool = True,
    progress_callback: ProgressCallback | None = None,
) -> list[dict[str, Any]]:
    """Stage 3 only: run AI evaluation for generated texts."""
    updated = deepcopy(results)
    runnable = [r for r in updated if r.get("text") and not r.get("error")]
    total = len(runnable)
    counter = 0
    for result in updated:
        if not result.get("text") or result.get("error"):
            continue
        current_eval = result.get("text_ai_eval", {})
        if current_eval and not current_eval.get("_skipped") and not overwrite:
            continue
        counter += 1
        label = f"{result.get('topic')} / {result.get('grade_label')}"
        if progress_callback:
            progress_callback("text_evaluation", label, counter, total)
        try:
            _set_model(result, "text_evaluator", text_evaluator)
            result["text_ai_eval"] = evaluate_text_ai(
                text_evaluator,
                result["text"].get("title", ""),
                result["text"].get("body", ""),
                result.get("params", result["text"].get("params", {})),
                result["text"].get("metrics", {}),
                prompt_overrides=prompt_overrides,
            )
        except Exception as exc:
            result["text_ai_eval"] = {"_overall_score": None, "_error": f"Text evaluation failed: {exc}"}
        refresh_approval(result, approval_threshold, min_dimension_score)
    return updated


def run_question_eval_stage(
    results: list[dict[str, Any]],
    question_evaluator: ModelSpec,
    approval_threshold: float = DEFAULT_APPROVAL_THRESHOLD,
    min_dimension_score: float = DEFAULT_MIN_DIMENSION_SCORE,
    prompt_overrides: dict[str, str] | None = None,
    overwrite: bool = True,
    progress_callback: ProgressCallback | None = None,
) -> list[dict[str, Any]]:
    """Stage 4 only: run AI evaluation for generated question sets."""
    updated = deepcopy(results)
    runnable = [r for r in updated if r.get("text") and r.get("questions") and not r.get("error")]
    total = len(runnable)
    counter = 0
    for result in updated:
        if not result.get("text") or not result.get("questions") or result.get("error"):
            continue
        current_eval = result.get("question_ai_eval", {})
        if current_eval and not current_eval.get("_skipped") and not overwrite:
            continue
        counter += 1
        label = f"{result.get('topic')} / {result.get('grade_label')}"
        if progress_callback:
            progress_callback("question_evaluation", label, counter, total)
        try:
            params = result.get("params", result["questions"].get("params", {}))
            _set_model(result, "question_evaluator", question_evaluator)
            result["question_ai_eval"] = evaluate_questions_ai(
                question_evaluator,
                result["text"].get("title", ""),
                strip_word_count_tag(result["text"].get("body", "")),
                result["questions"].get("questions", []),
                params,
                result["questions"].get("metrics", {}),
                prompt_overrides=prompt_overrides,
            )
        except Exception as exc:
            result["question_ai_eval"] = {"_overall_score": None, "_error": f"Question evaluation failed: {exc}"}
        refresh_approval(result, approval_threshold, min_dimension_score)
    return updated


def run_for_grade(
    text_generator: ModelSpec,
    question_generator: ModelSpec,
    text_evaluator: ModelSpec,
    question_evaluator: ModelSpec,
    topic: str,
    grade_key: str,
    word_count: int,
    question_count: int,
    text_type_key: str,
    text_format: str | None,
    approval_threshold: float = DEFAULT_APPROVAL_THRESHOLD,
    min_dimension_score: float = DEFAULT_MIN_DIMENSION_SCORE,
    word_tolerance: int = DEFAULT_WORD_TOLERANCE,
    max_revisions: int = DEFAULT_MAX_REVISIONS,
    run_text_ai_eval: bool = True,
    run_question_ai_eval: bool = True,
    temperature: float = 0.3,
    text_prompt_strategy: str = "few_shot",
    question_prompt_strategy: str = "few_shot",
    prompt_overrides: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Run all stages for one topic-grade item, with independent models per stage."""
    params = build_params(grade_key, topic, word_count, question_count, text_type_key, text_format)
    result = _base_result(
        topic,
        grade_key,
        params,
        text_generator=text_generator,
        question_generator=question_generator,
        text_evaluator=text_evaluator,
        question_evaluator=question_evaluator,
    )
    result["prompting"] = {
        "text_generation_strategy": text_prompt_strategy,
        "question_generation_strategy": question_prompt_strategy,
        "custom_prompts_used": bool(prompt_overrides),
    }

    result["text"] = generate_text(
        text_generator,
        params,
        word_tolerance=word_tolerance,
        max_revisions=max_revisions,
        temperature=temperature,
        prompt_strategy=text_prompt_strategy,
        prompt_overrides=prompt_overrides,
    )
    result["questions"] = generate_questions(
        question_generator,
        result["text"],
        temperature=min(temperature, 0.3),
        prompt_strategy=question_prompt_strategy,
        prompt_overrides=prompt_overrides,
    )

    if run_text_ai_eval:
        result["text_ai_eval"] = evaluate_text_ai(
            text_evaluator,
            result["text"].get("title", ""),
            result["text"].get("body", ""),
            params,
            result["text"].get("metrics", {}),
            prompt_overrides=prompt_overrides,
        )
    if run_question_ai_eval:
        result["question_ai_eval"] = evaluate_questions_ai(
            question_evaluator,
            result["text"].get("title", ""),
            strip_word_count_tag(result["text"].get("body", "")),
            result["questions"].get("questions", []),
            params,
            result["questions"].get("metrics", {}),
            prompt_overrides=prompt_overrides,
        )

    refresh_approval(result, approval_threshold, min_dimension_score)
    return result


def run_pipeline(
    text_generator: ModelSpec,
    question_generator: ModelSpec,
    text_evaluator: ModelSpec,
    question_evaluator: ModelSpec,
    topics: list[str],
    grade_keys: list[str],
    word_count: int,
    question_count: int,
    text_type_key: str = "mixed",
    text_format: str | None = None,
    approval_threshold: float = DEFAULT_APPROVAL_THRESHOLD,
    min_dimension_score: float = DEFAULT_MIN_DIMENSION_SCORE,
    word_tolerance: int = DEFAULT_WORD_TOLERANCE,
    max_revisions: int = DEFAULT_MAX_REVISIONS,
    run_text_ai_eval: bool = True,
    run_question_ai_eval: bool = True,
    temperature: float = 0.3,
    text_prompt_strategy: str = "few_shot",
    question_prompt_strategy: str = "few_shot",
    prompt_overrides: dict[str, str] | None = None,
    progress_callback: ProgressCallback | None = None,
) -> list[dict[str, Any]]:
    """Run the full four-stage pipeline with independent models for each stage."""
    results = run_text_stage(
        text_generator=text_generator,
        topics=topics,
        grade_keys=grade_keys,
        word_count=word_count,
        question_count=question_count,
        text_type_key=text_type_key,
        text_format=text_format,
        approval_threshold=approval_threshold,
        min_dimension_score=min_dimension_score,
        word_tolerance=word_tolerance,
        max_revisions=max_revisions,
        temperature=temperature,
        prompt_strategy=text_prompt_strategy,
        prompt_overrides=prompt_overrides,
        progress_callback=progress_callback,
    )
    results = run_question_stage(
        results=results,
        question_generator=question_generator,
        approval_threshold=approval_threshold,
        min_dimension_score=min_dimension_score,
        temperature=min(temperature, 0.3),
        prompt_strategy=question_prompt_strategy,
        prompt_overrides=prompt_overrides,
        progress_callback=progress_callback,
    )
    if run_text_ai_eval:
        results = run_text_eval_stage(
            results=results,
            text_evaluator=text_evaluator,
            approval_threshold=approval_threshold,
            min_dimension_score=min_dimension_score,
            prompt_overrides=prompt_overrides,
            progress_callback=progress_callback,
        )
    if run_question_ai_eval:
        results = run_question_eval_stage(
            results=results,
            question_evaluator=question_evaluator,
            approval_threshold=approval_threshold,
            min_dimension_score=min_dimension_score,
            prompt_overrides=prompt_overrides,
            progress_callback=progress_callback,
        )
    return results
