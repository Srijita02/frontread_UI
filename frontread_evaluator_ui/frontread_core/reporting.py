"""Save and export FrontRead UI run results."""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .metrics import flatten_ai_scores
from .pipeline import safe_slug

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "runs"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


def _as_dict(value: Any) -> dict[str, Any]:
    """Return value if it is a dict, otherwise return an empty dict."""
    return value if isinstance(value, dict) else {}


def _redact_model(model: dict[str, Any]) -> dict[str, Any]:
    clean = dict(model or {})
    if clean.get("api_key"):
        clean["api_key"] = "***redacted***"
    return clean


def _model(result: dict[str, Any], role: str) -> dict[str, Any]:
    """Return a public model dict for a role, with fallbacks for older result files."""
    models = _as_dict(result.get("models"))
    if role in models:
        return _as_dict(models.get(role))
    if role == "text_generator":
        return _as_dict(result.get("generator"))
    if role in {"text_evaluator", "question_evaluator"}:
        return _as_dict(result.get("evaluator"))
    if role == "question_generator":
        return _as_dict(result.get("generator"))
    return {}


def _extract_dimension_score(eval_dict: dict[str, Any], key: str) -> int | float | None:
    """Support both {'score': n} dimensions and direct numeric dimensions."""
    eval_dict = _as_dict(eval_dict)
    value = eval_dict.get(key)

    if isinstance(value, dict):
        score = value.get("score")
        return score if isinstance(score, (int, float)) else None

    if isinstance(value, (int, float)):
        return value

    return None


def _overall_score(eval_dict: dict[str, Any]) -> int | float | None:
    """
    Support old and new evaluation formats.

    Old:
      {"_overall_score": 4}

    New:
      {"overall_quality": {"score": 4, "justification": "..."}}
    """
    eval_dict = _as_dict(eval_dict)

    old_score = eval_dict.get("_overall_score")
    if isinstance(old_score, (int, float)):
        return old_score

    return _extract_dimension_score(eval_dict, "overall_quality")


def _min_score(eval_dict: dict[str, Any]) -> int | float | None:
    """Minimum score across scored dimensions, excluding overall fields."""
    eval_dict = _as_dict(eval_dict)
    scores: list[int | float] = []

    for key, value in eval_dict.items():
        if key in {"_overall_score", "overall_quality"}:
            continue

        if isinstance(value, dict):
            score = value.get("score")
            if isinstance(score, (int, float)):
                scores.append(score)
        elif isinstance(value, (int, float)):
            scores.append(value)

    return min(scores) if scores else None


def _metric_bool(value: Any) -> bool:
    """Treat only True as passing."""
    return value is True


def _approval_bool(value: Any) -> bool | None:
    """Return bool if available, otherwise None."""
    return value if isinstance(value, bool) else None


def serialisable_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    clean_results = []
    for result in results:
        clone = json.loads(json.dumps(result, ensure_ascii=False, default=str))
        if isinstance(clone.get("generator"), dict):
            clone["generator"] = _redact_model(clone["generator"])
        if isinstance(clone.get("evaluator"), dict):
            clone["evaluator"] = _redact_model(clone["evaluator"])
        if isinstance(clone.get("models"), dict):
            clone["models"] = {role: _redact_model(model) for role, model in clone["models"].items()}
        clean_results.append(clone)
    return clean_results


def summary_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []

    for result in results:
        result = _as_dict(result)
        text = _as_dict(result.get("text"))
        questions = _as_dict(result.get("questions"))
        tm = _as_dict(text.get("metrics"))
        qm = _as_dict(questions.get("metrics"))
        approval = _as_dict(result.get("approval"))

        text_ai_eval = _as_dict(result.get("text_ai_eval"))
        question_ai_eval = _as_dict(result.get("question_ai_eval"))

        text_gen = _model(result, "text_generator")
        question_gen = _model(result, "question_generator")
        text_eval = _model(result, "text_evaluator")
        question_eval = _model(result, "question_evaluator")
        prompting = _as_dict(result.get("prompting"))

        text_ai_overall_score = approval.get("text_ai_overall_score")
        if text_ai_overall_score is None:
            text_ai_overall_score = _overall_score(text_ai_eval)

        question_ai_overall_score = approval.get("question_ai_overall_score")
        if question_ai_overall_score is None:
            question_ai_overall_score = _overall_score(question_ai_eval)

        text_ai_min_score = approval.get("text_ai_min_score")
        if text_ai_min_score is None:
            text_ai_min_score = _min_score(text_ai_eval)

        question_ai_min_score = approval.get("question_ai_min_score")
        if question_ai_min_score is None:
            question_ai_min_score = _min_score(question_ai_eval)

        text_metric_ok = approval.get("text_metric_ok")
        if text_metric_ok is None:
            text_metric_ok = _metric_bool(tm.get("word_count_ok")) and _metric_bool(tm.get("lix_in_band"))

        question_metric_ok = approval.get("question_metric_ok")
        if question_metric_ok is None:
            question_metric_ok = (
                _metric_bool(qm.get("question_count_ok"))
                and _metric_bool(qm.get("questions_complete_ok"))
                and _metric_bool(qm.get("text_references_ok"))
            )

        text_approved = _approval_bool(approval.get("text_approved"))
        if text_approved is None and text_ai_overall_score is not None and text_ai_min_score is not None:
            text_approved = bool(text_metric_ok and text_ai_overall_score >= 4 and text_ai_min_score >= 3)

        questions_approved = _approval_bool(approval.get("questions_approved"))
        if (
            questions_approved is None
            and question_ai_overall_score is not None
            and question_ai_min_score is not None
        ):
            questions_approved = bool(
                question_metric_ok
                and question_ai_overall_score >= 4
                and question_ai_min_score >= 3
            )

        overall_approved = _approval_bool(approval.get("overall_approved"))
        if overall_approved is None and text_approved is not None and questions_approved is not None:
            overall_approved = bool(text_approved and questions_approved)

        row = {
            "run_item_id": result.get("run_item_id"),
            "created_at": result.get("created_at"),
            "topic": result.get("topic"),
            "grade": result.get("grade_label"),
            "grade_key": result.get("grade_key"),
            "title": text.get("title", ""),
            "text_gen_provider": text_gen.get("provider", ""),
            "text_gen_model_id": text_gen.get("model", ""),
            "question_gen_provider": question_gen.get("provider", ""),
            "question_gen_model_id": question_gen.get("model", ""),
            "text_eval_provider": text_eval.get("provider", ""),
            "text_eval_model_id": text_eval.get("model", ""),
            "question_eval_provider": question_eval.get("provider", ""),
            "question_eval_model_id": question_eval.get("model", ""),
            "text_prompt_strategy": prompting.get("text_generation_strategy", ""),
            "question_prompt_strategy": prompting.get("question_generation_strategy", ""),
            "custom_prompts_used": prompting.get("custom_prompts_used", False),

            # Backwards-compatible names used by earlier notebooks/exports.
            "model_provider": text_gen.get("provider", ""),
            "model_id": text_gen.get("model", ""),
            "eval_provider": text_eval.get("provider", ""),
            "eval_model_id": text_eval.get("model", ""),

            # Text metrics.
            "word_count_target": tm.get("target_word_count"),
            "word_count_actual": tm.get("actual_word_count"),
            "word_count_diff": tm.get("word_count_diff"),
            "word_count_ok": tm.get("word_count_ok"),
            "lix_target": tm.get("lix_target"),
            "lix_min": tm.get("lix_min"),
            "lix_max": tm.get("lix_max"),
            "lix_actual": tm.get("lix_actual"),
            "lix_diff": tm.get("lix_diff"),
            "lix_in_band": tm.get("lix_in_band"),
            "avg_sentence_length": tm.get("avg_sentence_length"),
            "long_word_pct": tm.get("long_word_pct"),
            "paragraph_lix_values": ", ".join(str(x) for x in tm.get("paragraph_lix_values", [])),

            # Question metrics.
            "question_count_target": qm.get("question_count_target"),
            "question_count_actual": qm.get("question_count_actual"),
            "question_count_ok": qm.get("question_count_ok"),
            "questions_complete_ok": qm.get("questions_complete_ok"),

            # AI scores and approval.
            "text_ai_overall_score": text_ai_overall_score,
            "text_ai_min_score": text_ai_min_score,
            "question_ai_overall_score": question_ai_overall_score,
            "question_ai_min_score": question_ai_min_score,
            "text_metric_ok": text_metric_ok,
            "question_metric_ok": question_metric_ok,
            "text_approved": text_approved,
            "questions_approved": questions_approved,
            "overall_approved": overall_approved,

            "error": result.get("error", "") or result.get("question_error", ""),
        }

        rows.append(row)

    return rows


def text_metric_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []

    for result in results:
        result = _as_dict(result)
        params = _as_dict(result.get("params"))
        text = _as_dict(result.get("text"))
        tm = _as_dict(text.get("metrics"))
        text_gen = _model(result, "text_generator")
        text_eval = _model(result, "text_evaluator")
        prompting = _as_dict(result.get("prompting"))

        row = {
            "run_item_id": result.get("run_item_id"),
            "created_at": result.get("created_at"),
            "topic": result.get("topic"),
            "grade": result.get("grade_label"),
            "grade_key": result.get("grade_key"),
            "source_file": result.get("source_file", ""),
            "model_id": text_gen.get("model", ""),
            "model_name": text_gen.get("display_name", ""),
            "eval_model_id": text_eval.get("model", ""),
            "eval_model_name": text_eval.get("display_name", ""),
            "text_prompt_strategy": prompting.get("text_generation_strategy", ""),
            "custom_prompts_used": prompting.get("custom_prompts_used", False),
            "text_type": params.get("text_type_label"),
            "text_format": params.get("text_format"),
            "title": text.get("title"),
            "word_count_target": tm.get("target_word_count"),
            "word_count_actual": tm.get("actual_word_count"),
            "word_count_diff": tm.get("word_count_diff"),
            "word_count_ok": tm.get("word_count_ok"),
            "total_sentences": tm.get("total_sentences"),
            "avg_sentence_length": tm.get("avg_sentence_length"),
            "long_word_count": tm.get("long_word_count"),
            "long_word_pct": tm.get("long_word_pct"),
            "lix_target": tm.get("lix_target"),
            "lix_min": tm.get("lix_min"),
            "lix_max": tm.get("lix_max"),
            "lix_actual": tm.get("lix_actual"),
            "lix_diff": tm.get("lix_diff"),
            "lix_in_band": tm.get("lix_in_band"),
            "paragraph_lix_scores": json.dumps(tm.get("paragraph_lix_scores", []), ensure_ascii=False),
            "revisions_needed": text.get("final_revision"),
            "elapsed_sec": text.get("elapsed_sec"),
        }

        row.update(flatten_ai_scores(_as_dict(result.get("text_ai_eval")), "text_ai"))
        rows.append(row)

    return rows


def question_metric_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []

    for result in results:
        result = _as_dict(result)
        questions = _as_dict(result.get("questions"))
        qm = _as_dict(questions.get("metrics"))
        question_gen = _model(result, "question_generator")
        question_eval = _model(result, "question_evaluator")
        prompting = _as_dict(result.get("prompting"))

        row = {
            "run_item_id": result.get("run_item_id"),
            "created_at": result.get("created_at"),
            "topic": result.get("topic"),
            "grade": result.get("grade_label"),
            "grade_key": result.get("grade_key"),
            "title": questions.get("title"),
            "question_model_id": question_gen.get("model", ""),
            "question_model_name": question_gen.get("display_name", ""),
            "eval_model_id": question_eval.get("model", ""),
            "eval_model_name": question_eval.get("display_name", ""),
            "question_prompt_strategy": prompting.get("question_generation_strategy", ""),
            "custom_prompts_used": prompting.get("custom_prompts_used", False),
            "question_count_target": qm.get("question_count_target"),
            "question_count_actual": qm.get("question_count_actual"),
            "question_count_ok": qm.get("question_count_ok"),
            "type_distribution": qm.get("type_distribution"),
            "answer_distribution": qm.get("answer_distribution"),
            "questions_with_4_options": qm.get("questions_with_4_options"),
            "questions_complete_ok": qm.get("questions_complete_ok"),
            "questions_with_text_reference": qm.get("questions_with_text_reference"),
            "text_references_ok": qm.get("text_references_ok"),
            "elapsed_sec": questions.get("elapsed_sec"),
        }

        row.update(flatten_ai_scores(_as_dict(result.get("question_ai_eval")), "question_ai"))
        rows.append(row)

    return rows


def question_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []

    for result in results:
        result = _as_dict(result)
        questions_obj = _as_dict(result.get("questions"))
        text_obj = _as_dict(result.get("text"))

        for q in questions_obj.get("questions", []) or []:
            q = _as_dict(q)
            rows.append(
                {
                    "run_item_id": result.get("run_item_id"),
                    "topic": result.get("topic"),
                    "grade": result.get("grade_label"),
                    "title": text_obj.get("title"),
                    "number": q.get("number"),
                    "question": q.get("question"),
                    "A": q.get("A"),
                    "B": q.get("B"),
                    "C": q.get("C"),
                    "D": q.get("D"),
                    "correct": q.get("correct"),
                    "text_reference": q.get("text_reference"),
                }
            )

    return rows


def save_run(results: list[dict[str, Any]], run_config: dict[str, Any] | None = None) -> dict[str, Path]:
    """Persist results as JSON, CSV, XLSX, text files, and a ZIP archive."""
    run_config = run_config or {}
    topic_slug = safe_slug("_".join(sorted({r.get("topic", "topic") for r in results}) or ["topic"]))
    run_id = run_config.get("run_id") or f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{topic_slug}"
    run_dir = OUTPUT_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    clean = serialisable_results(results)

    (run_dir / "results.json").write_text(
        json.dumps(clean, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    (run_dir / "run_config.json").write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    summary_df = pd.DataFrame(summary_rows(clean))
    text_df = pd.DataFrame(text_metric_rows(clean))
    question_eval_df = pd.DataFrame(question_metric_rows(clean))
    questions_df = pd.DataFrame(question_rows(clean))

    summary_path = run_dir / "summary.csv"
    summary_df.to_csv(summary_path, index=False)

    xlsx_path = run_dir / "frontread_evaluation_report.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        text_df.to_excel(writer, sheet_name="Text Metrics + AI", index=False)
        question_eval_df.to_excel(writer, sheet_name="Question Metrics + AI", index=False)
        questions_df.to_excel(writer, sheet_name="Questions", index=False)

    texts_dir = run_dir / "generated_texts"
    texts_dir.mkdir(exist_ok=True)

    for idx, result in enumerate(clean, start=1):
        result = _as_dict(result)
        text = _as_dict(result.get("text"))
        params = _as_dict(result.get("params"))
        approval = _as_dict(result.get("approval"))

        fname = f"{idx:03d}_{safe_slug(result.get('topic', 'topic'))}_{params.get('grade_key', 'grade')}.md"

        content = f"# {text.get('title', 'Untitled')}\n\n"
        content += f"Topic: {result.get('topic')}\n\n"
        content += f"Grade: {result.get('grade_label')}\n\n"
        content += f"Text approved: {approval.get('text_approved')}\n\n"
        content += text.get("body", "") + "\n\n## Questions\n\n"

        questions_obj = _as_dict(result.get("questions"))
        for q in questions_obj.get("questions", []) or []:
            q = _as_dict(q)
            content += f"{q.get('number')}. {q.get('question')}\n"
            for letter in "ABCD":
                content += f"{letter}: {q.get(letter, '')}\n"
            content += f"Correct: {q.get('correct')}\n"
            content += f"Text reference: {q.get('text_reference', '')}\n\n"

        (texts_dir / fname).write_text(content, encoding="utf-8")

    zip_base = str(run_dir)
    zip_path = shutil.make_archive(zip_base, "zip", root_dir=run_dir)

    return {
        "run_dir": run_dir,
        "json": run_dir / "results.json",
        "csv": summary_path,
        "xlsx": xlsx_path,
        "zip": Path(zip_path),
    }