"""Deterministic evaluation metrics for FrontRead texts and questions."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

from .config import DEFAULT_WORD_TOLERANCE

_WORD_RE = re.compile(r"\b[\w']+\b", re.UNICODE)
_SENTENCE_SPLIT_RE = re.compile(r"[.!?]+(?:\s|$)")


def strip_word_count_tag(text: str) -> str:
    """Remove a model-declared word-count tag, including optional LIX/grade metadata."""
    return re.sub(r"\[\s*Word count\s*:[^\]]*\]", "", text or "", flags=re.IGNORECASE).strip()


def count_words(text: str) -> int:
    return len(_WORD_RE.findall(text or ""))


def count_sentences(text: str) -> int:
    clean = (text or "").strip()
    if not clean:
        return 0
    parts = [s.strip() for s in _SENTENCE_SPLIT_RE.split(clean) if s.strip()]
    return max(1, len(parts))


def count_long_words(text: str) -> int:
    return sum(1 for w in _WORD_RE.findall(text or "") if len(w.strip("'")) > 6)


def compute_lix(text: str) -> float:
    clean = strip_word_count_tag(text)
    words = count_words(clean)
    sentences = count_sentences(clean)
    long_words = count_long_words(clean)
    if words <= 0 or sentences <= 0:
        return 0.0
    return round((words / sentences) + (long_words * 100 / words), 1)


def paragraph_lix_scores(text: str) -> list[dict[str, Any]]:
    clean = strip_word_count_tag(text)
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", clean) if p.strip()]
    return [
        {
            "paragraph": i + 1,
            "lix": compute_lix(paragraph),
            "word_count": count_words(paragraph),
            "preview": paragraph[:120] + ("..." if len(paragraph) > 120 else ""),
        }
        for i, paragraph in enumerate(paragraphs)
    ]


def declared_word_count(text: str) -> int | None:
    match = re.search(r"\[\s*Word count\s*:\s*(\d+)", text or "", flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def text_metrics(body: str, params: dict[str, Any], word_tolerance: int = DEFAULT_WORD_TOLERANCE) -> dict[str, Any]:
    clean = strip_word_count_tag(body)
    actual_words = count_words(clean)
    target_words = int(params.get("word_count", 0) or 0)
    total_sentences = count_sentences(clean)
    long_word_count = count_long_words(clean)
    long_word_pct = round((long_word_count * 100 / actual_words), 1) if actual_words else 0.0
    avg_sentence_length = round((actual_words / total_sentences), 1) if total_sentences else 0.0
    lix_actual = compute_lix(clean)
    lix_target = float(params.get("lix_target", 0) or 0)
    lix_min = float(params.get("lix_min", 0) or 0)
    lix_max = float(params.get("lix_max", 100) or 100)
    word_count_diff = actual_words - target_words
    p_scores = paragraph_lix_scores(clean)
    p_lix_values = [p["lix"] for p in p_scores]
    p_lix_variance = round(max(p_lix_values) - min(p_lix_values), 1) if p_lix_values else 0.0

    return {
        "target_word_count": target_words,
        "actual_word_count": actual_words,
        "declared_word_count": declared_word_count(body),
        "word_count_diff": word_count_diff,
        "word_count_ok": abs(word_count_diff) <= int(word_tolerance),
        "word_count_tolerance": int(word_tolerance),
        "total_sentences": total_sentences,
        "avg_sentence_length": avg_sentence_length,
        "long_word_count": long_word_count,
        "long_word_pct": long_word_pct,
        "lix_target": lix_target,
        "lix_min": lix_min,
        "lix_max": lix_max,
        "lix_actual": lix_actual,
        "lix_diff": round(lix_actual - lix_target, 1),
        "lix_in_band": lix_min <= lix_actual <= lix_max,
        "paragraph_lix_scores": p_scores,
        "paragraph_lix_values": p_lix_values,
        "paragraph_lix_variance": p_lix_variance,
    }


def describe_text_metric_problems(metrics: dict[str, Any]) -> str:
    problems: list[str] = []
    if not metrics.get("word_count_ok", False):
        diff = int(metrics.get("word_count_diff", 0))
        direction = "too long" if diff > 0 else "too short"
        problems.append(
            f"Word count is {diff:+d} ({direction}). Target is {metrics.get('target_word_count')} ±{metrics.get('word_count_tolerance', 20)}."
        )
    if not metrics.get("lix_in_band", False):
        actual = metrics.get("lix_actual")
        lo = metrics.get("lix_min")
        hi = metrics.get("lix_max")
        direction = "too high" if float(actual) > float(hi) else "too low"
        action = "shorten sentences and use simpler words" if direction == "too high" else "use slightly longer sentences and more words over six characters"
        problems.append(f"LIX is {actual}, which is {direction}. Required range is {lo}–{hi}; {action}.")
    if float(metrics.get("paragraph_lix_variance", 0)) > 10:
        problems.append(
            f"Paragraph LIX values vary too much ({metrics.get('paragraph_lix_values')}). Keep paragraph difficulty more consistent."
        )
    if not problems:
        problems.append("No major metric violations. Make only light polishing changes if needed.")
    return "\n".join(f"- {p}" for p in problems)


def question_metrics(questions: list[dict[str, Any]], target_count: int) -> dict[str, Any]:
    q_actual = len(questions)
    q_target = int(target_count or 0)
    answer_counts = Counter((q.get("correct") or "").upper() for q in questions if q.get("correct"))
    type_counts = Counter(q.get("type") or "Literal" for q in questions)

    def complete(q: dict[str, Any]) -> bool:
        return bool(q.get("question")) and all(bool(q.get(letter)) for letter in "ABCD") and (q.get("correct") in list("ABCD"))

    complete_count = sum(1 for q in questions if complete(q))
    refs_count = sum(1 for q in questions if q.get("text_reference"))
    answer_distribution = ", ".join(f"{letter}:{answer_counts.get(letter, 0)}" for letter in "ABCD")
    type_distribution = ", ".join(f"{k}:{v}" for k, v in sorted(type_counts.items())) or ""

    return {
        "question_count_target": q_target,
        "question_count_actual": q_actual,
        "question_count_ok": q_actual == q_target,
        "type_distribution": type_distribution,
        "answer_distribution": answer_distribution,
        "questions_with_4_options": complete_count,
        "questions_complete_ok": complete_count == q_actual and q_actual > 0,
        "questions_with_text_reference": refs_count,
        "text_references_ok": refs_count == q_actual and q_actual > 0,
    }


def flatten_ai_scores(ai_eval: dict[str, Any], prefix: str) -> dict[str, Any]:
    """Flatten {dimension: {score, justification}} into report-friendly columns."""
    flat: dict[str, Any] = {}
    for dim, payload in (ai_eval or {}).items():
        if dim.startswith("_"):
            continue
        if isinstance(payload, dict):
            flat[f"{prefix}_{dim}_score"] = payload.get("score")
            flat[f"{prefix}_{dim}_justification"] = payload.get("justification")
    flat[f"{prefix}_overall_score"] = ai_overall_score(ai_eval)
    return flat


def ai_overall_score(ai_eval: dict[str, Any]) -> float | None:
    scores: list[float] = []
    for dim, payload in (ai_eval or {}).items():
        if dim.startswith("_") or not isinstance(payload, dict):
            continue
        try:
            score = float(payload.get("score"))
            if not math.isnan(score):
                scores.append(score)
        except (TypeError, ValueError):
            continue
    return round(sum(scores) / len(scores), 2) if scores else None


def ai_min_score(ai_eval: dict[str, Any]) -> float | None:
    scores: list[float] = []
    for dim, payload in (ai_eval or {}).items():
        if dim.startswith("_") or not isinstance(payload, dict):
            continue
        try:
            score = float(payload.get("score"))
            if not math.isnan(score):
                scores.append(score)
        except (TypeError, ValueError):
            continue
    return min(scores) if scores else None


def approval_flags(
    text_metric_row: dict[str, Any],
    question_metric_row: dict[str, Any],
    text_ai_eval: dict[str, Any],
    question_ai_eval: dict[str, Any],
    approval_threshold: float,
    min_dimension_score: float,
) -> dict[str, Any]:
    text_ai_avg = ai_overall_score(text_ai_eval)
    question_ai_avg = ai_overall_score(question_ai_eval)
    text_ai_min = ai_min_score(text_ai_eval)
    question_ai_min = ai_min_score(question_ai_eval)

    text_metric_ok = bool(text_metric_row.get("word_count_ok")) and bool(text_metric_row.get("lix_in_band"))
    text_ai_ok = text_ai_avg is not None and text_ai_avg >= approval_threshold and (text_ai_min or 0) >= min_dimension_score
    question_metric_ok = bool(question_metric_row.get("question_count_ok")) and bool(question_metric_row.get("questions_complete_ok"))
    question_ai_ok = question_ai_avg is not None and question_ai_avg >= approval_threshold and (question_ai_min or 0) >= min_dimension_score

    return {
        "text_metric_ok": text_metric_ok,
        "text_ai_ok": text_ai_ok,
        "text_approved": text_metric_ok and text_ai_ok,
        "question_metric_ok": question_metric_ok,
        "question_ai_ok": question_ai_ok,
        "questions_approved": question_metric_ok and question_ai_ok,
        "overall_approved": text_metric_ok and text_ai_ok and question_metric_ok and question_ai_ok,
        "text_ai_overall_score": text_ai_avg,
        "text_ai_min_score": text_ai_min,
        "question_ai_overall_score": question_ai_avg,
        "question_ai_min_score": question_ai_min,
    }
