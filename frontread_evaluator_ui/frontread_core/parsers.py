"""Robust output parsing utilities for model responses."""

from __future__ import annotations

import json
import re
from typing import Any


def parse_text_output(raw: str) -> tuple[str, str]:
    """Parse a model response into (title, body). Tolerates minor formatting drift."""
    raw = (raw or "").strip()
    if not raw:
        return "", ""

    title = ""
    body = ""

    title_match = re.search(r"Title\s*:\s*\n\s*(.+?)(?=\n)", raw, flags=re.IGNORECASE | re.DOTALL)
    if title_match:
        title = title_match.group(1).strip().strip("# ")
    else:
        lines = [line.strip() for line in raw.splitlines() if line.strip()]
        title = re.sub(r"^#+\s*", "", lines[0]) if lines else "Untitled"

    body_match = re.search(r"Body text\s*:\s*\n(?P<body>.*)\Z", raw, flags=re.IGNORECASE | re.DOTALL)
    if body_match:
        body = body_match.group("body").strip()
    else:
        # If the body header is missing, remove the first non-empty line and keep the rest.
        lines = raw.splitlines()
        body_lines: list[str] = []
        skipped_title = False
        for line in lines:
            if not skipped_title and line.strip():
                skipped_title = True
                continue
            body_lines.append(line)
        body = "\n".join(body_lines).strip()

    body = re.sub(r"^#+\s*", "", body, flags=re.MULTILINE).strip()
    return title or "Untitled", body


def _field(block: str, label: str, next_labels: list[str]) -> str:
    next_part = "|".join(re.escape(n) for n in next_labels)
    pattern = rf"(?:^|\n)\s*{re.escape(label)}\s*[:\).\-]\s*(.*?)(?=\n\s*(?:{next_part})\s*[:\).\-]|\Z)"
    match = re.search(pattern, block, flags=re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else ""


def parse_questions_output(raw: str) -> list[dict[str, Any]]:
    """Parse numbered multiple-choice questions with A-D options, correct answer, and text reference."""
    raw = (raw or "").strip()
    if not raw:
        return []

    blocks = re.findall(r"(?ms)^\s*(\d+)\s*[\.)]\s*(.*?)(?=^\s*\d+\s*[\.)]\s|\Z)", raw)
    questions: list[dict[str, Any]] = []

    for fallback_number, block in blocks:
        a_pos = re.search(r"(?:^|\n)\s*A\s*[:\).\-]", block, flags=re.IGNORECASE)
        question_text = block[: a_pos.start()].strip() if a_pos else block.strip()
        question_text = re.sub(r"^Question\s*[:\-]\s*", "", question_text, flags=re.IGNORECASE).strip()
        question_text = " ".join(question_text.split())

        item = {
            "number": int(fallback_number),
            "question": question_text,
            "A": _field(block, "A", ["B", "C", "D", "Correct", "Text reference", "Type"]),
            "B": _field(block, "B", ["C", "D", "Correct", "Text reference", "Type"]),
            "C": _field(block, "C", ["D", "Correct", "Text reference", "Type"]),
            "D": _field(block, "D", ["Correct", "Text reference", "Type"]),
            "correct": "",
            "text_reference": "",
            "type": "Literal",
        }
        correct_match = re.search(r"(?:^|\n)\s*Correct\s*[:\-]\s*([ABCD])\b", block, flags=re.IGNORECASE)
        if correct_match:
            item["correct"] = correct_match.group(1).upper()
        ref_match = re.search(r"(?:^|\n)\s*Text reference\s*[:\-]\s*(.*?)(?=\n\s*Type\s*[:\-]|\Z)", block, flags=re.IGNORECASE | re.DOTALL)
        if ref_match:
            item["text_reference"] = " ".join(ref_match.group(1).strip().split())
        type_match = re.search(r"(?:^|\n)\s*Type\s*[:\-]\s*(.+?)\s*$", block, flags=re.IGNORECASE | re.DOTALL)
        if type_match:
            item["type"] = type_match.group(1).strip()
        questions.append(item)

    return questions


def extract_json_object(raw: str) -> dict[str, Any]:
    """Extract the first JSON object from a model response, with light markdown cleanup."""
    text = (raw or "").strip()
    if not text:
        raise ValueError("No JSON content returned by evaluator model.")
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    if start < 0:
        raise ValueError("No JSON object start found in evaluator output.")

    depth = 0
    in_string = False
    escaped = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : idx + 1]
                return json.loads(candidate)

    # Last resort: close missing braces.
    candidate = text[start:] + ("}" * max(0, depth))
    return json.loads(candidate)


def normalise_eval_payload(payload: dict[str, Any], dimensions: list[str]) -> dict[str, Any]:
    """Ensure each requested dimension has a {score, justification} object."""
    normalised: dict[str, Any] = {}
    for dim in dimensions:
        value = payload.get(dim, {}) if isinstance(payload, dict) else {}
        if isinstance(value, (int, float, str)):
            value = {"score": value, "justification": ""}
        score = value.get("score") if isinstance(value, dict) else None
        try:
            score = int(round(float(score)))
        except (TypeError, ValueError):
            score = None
        if score is not None:
            score = max(1, min(5, score))
        normalised[dim] = {
            "score": score,
            "justification": str(value.get("justification", "")) if isinstance(value, dict) else "",
        }
    return normalised
