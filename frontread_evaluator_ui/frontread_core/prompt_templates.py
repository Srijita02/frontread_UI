"""Prompt strategy templates and rendering helpers for the FrontRead UI.

The UI can expose these templates for viewing/editing. The templates use a
small Jinja-style placeholder syntax, for example {{grade_label}} or {{body}}.
This avoids Python .format() collisions with JSON snippets or rubric braces.
"""

from __future__ import annotations

import json
from typing import Any

from .config import QUESTION_EVAL_DIMS, TEXT_EVAL_DIMS
from .prompts_fewshot import (
    EXEMPLAR_QA_HIGH,
    EXEMPLAR_QA_LOW,
    EXEMPLAR_QA_MID,
    EXEMPLAR_TEXT_HIGH,
    EXEMPLAR_TEXT_LOW,
    EXEMPLAR_TEXT_MID,
    TEXT_REVISION_PROMPT,
)

PROMPT_STRATEGIES = {
    "zero_shot": "Zero-shot",
    "one_shot": "One-shot",
    "few_shot": "Few-shot",
}

TEXT_GENERATION_SYSTEM_TEMPLATE = """
You are an expert educational writer for FrontRead, a Scandinavian reading-training platform.
Write in British English throughout (colour, organise, centre, practise, maths, etc.).

You write age-appropriate reading texts for school students. Your writing style matches a children's author: clear, engaging, with a strong narrative arc even in non-fiction.

STRICT RULES:
- British English only. No American spellings.
- Plain prose only inside the body — no bullet points, no bold, no markdown headers.
- Do NOT introduce more than one new concept or term per text. This is a reading exercise, not a lesson.
- Every paragraph must maintain a comparable readability level to the whole text.
- Write as close as possible to the requested word count and stay within the stated tolerance.
- Output ONLY the structured text in the required format. Nothing else.
""".strip()

TEXT_GENERATION_USER_ZERO_TEMPLATE = """
Generate a new FrontRead reading text.

PARAMETERS:
- Grade: {{grade_label}}
- Topic: {{topic}}
- Text type: {{text_type}} ({{text_format}})
- Target word count: EXACTLY {{word_count}} words (±20 words tolerance)
- Target LIX score: {{lix_target}} (acceptable range: {{lix_min}}–{{lix_max}})
- Avg sentence length to aim for: ~{{sentence_length_avg}} words per sentence
- Vocabulary: {{vocabulary_notes}}
- British English throughout

LIX FORMULA: LIX = (total_words ÷ total_sentences) + (long_words × 100 ÷ total_words)
Long words = words with MORE than 6 characters.
To INCREASE LIX: longer sentences, more words with >6 characters.
To DECREASE LIX: shorter sentences, simpler words (≤6 characters).

NEW CONCEPTS: At most one new or unfamiliar term per text.
PARAGRAPH CONSISTENCY: Every paragraph must have a similar LIX level.

OUTPUT FORMAT — use these exact headers:

Title:
[title here]

Body text:
[Word count: NNN]
[Full text here — plain prose only]
""".strip()

TEXT_GENERATION_USER_ONE_TEMPLATE = """
Below is one example of a correctly written FrontRead reading text. Study the format, paragraph style, readability control, and British English usage.

=== EXAMPLE: Medium difficulty (Grade 6, LIX ~22, 392 words) ===

{{exemplar_mid}}

===

Now generate a NEW reading text following the same format.
Do NOT copy or closely paraphrase the example. Write entirely original content on the new topic.

PARAMETERS:
- Grade: {{grade_label}}
- Topic: {{topic}}
- Text type: {{text_type}} ({{text_format}})
- Target word count: EXACTLY {{word_count}} words (±20 words tolerance)
- Target LIX score: {{lix_target}} (acceptable range: {{lix_min}}–{{lix_max}})
- Avg sentence length to aim for: ~{{sentence_length_avg}} words per sentence
- Vocabulary: {{vocabulary_notes}}
- British English throughout

LIX FORMULA: LIX = (total_words ÷ total_sentences) + (long_words × 100 ÷ total_words)
Long words = words with MORE than 6 characters.
To INCREASE LIX: longer sentences, more words with >6 characters.
To DECREASE LIX: shorter sentences, simpler words (≤6 characters).

NEW CONCEPTS: At most one new or unfamiliar term per text.
PARAGRAPH CONSISTENCY: Every paragraph must have a similar LIX level.

OUTPUT FORMAT — use these exact headers:

Title:
[title here]

Body text:
[Word count: NNN]
[Full text here — plain prose only]
""".strip()

TEXT_GENERATION_USER_FEW_TEMPLATE = """
Below are three examples of correctly written reading texts at different difficulty levels.
Study them carefully. Notice how:
- Vocabulary and sentence length increase from the low to the high example
- Each paragraph stays at a comparable difficulty level throughout the text
- The [Word count: NNN] tag appears as the first line of the body
- Plain prose only — no bullet points, no bold, no headers inside the text
- British English is used throughout

=== EXAMPLE 1: Low difficulty (Grade 3, LIX ~13, 214 words) ===

{{exemplar_low}}

=== EXAMPLE 2: Medium difficulty (Grade 6, LIX ~22, 392 words) ===

{{exemplar_mid}}

=== EXAMPLE 3: High difficulty (Grade 9, LIX ~34, 396 words) ===

{{exemplar_high}}

===

Now generate a NEW reading text following the same format.
Do NOT copy or closely paraphrase any example. Write entirely original content on the new topic.

PARAMETERS:
- Grade: {{grade_label}}
- Topic: {{topic}}
- Text type: {{text_type}} ({{text_format}})
- Target word count: EXACTLY {{word_count}} words (±20 words tolerance)
- Target LIX score: {{lix_target}} (acceptable range: {{lix_min}}–{{lix_max}})
- Avg sentence length to aim for: ~{{sentence_length_avg}} words per sentence
- Vocabulary: {{vocabulary_notes}}
- British English throughout

LIX FORMULA: LIX = (total_words ÷ total_sentences) + (long_words × 100 ÷ total_words)
Long words = words with MORE than 6 characters.
To INCREASE LIX: longer sentences, more words with >6 characters.
To DECREASE LIX: shorter sentences, simpler words (≤6 characters).

NEW CONCEPTS: At most one new or unfamiliar term per text.
PARAGRAPH CONSISTENCY: Every paragraph must have a similar LIX level.

OUTPUT FORMAT — use these exact headers:

Title:
[title here]

Body text:
[Word count: NNN]
[Full text here — plain prose only]
""".strip()

QUESTION_GENERATION_SYSTEM_TEMPLATE = """
You are an expert question writer for FrontRead, a Scandinavian reading-training platform.
Write in British English throughout.

You write LITERAL COMPREHENSION questions only. Every question must be directly and explicitly answered somewhere in the text.

ABSOLUTE RULES — no exceptions:
1. Every answer must be stated explicitly in the text. No inference. No reading between lines.
2. The correct answer must be a fact, name, number, place, or action written in the text.
3. Questions must NOT be answerable from general knowledge alone.
4. Questions must NOT be opinion-based.
5. Questions must NOT require analysis, inference, or critical thinking.
6. All four options (A, B, C, D) must be plausible but only ONE is correct per the text.
7. Wrong options must not be obviously silly.
8. Never use "all of the above" or "none of the above".
9. Questions must follow the order of the text.
10. British English throughout.
Output ONLY the structured questions. Nothing else.
""".strip()

QUESTION_GENERATION_USER_ZERO_TEMPLATE = """
Generate {{question_count}} literal comprehension questions for the text below.

TEXT DETAILS:
- Grade: {{grade_label}}
- Topic: {{topic}}
- Text word count: {{word_count}}

THE TEXT:
Title: {{title}}

{{body}}

REQUIREMENTS:
- Exactly {{question_count}} questions
- ALL questions are literal — answer must be explicitly stated in the text
- No inference, no opinion, no general knowledge
- Questions follow the order of the text
- One unambiguous correct answer per question
- Three plausible but wrong distractors
- Text reference quotes the exact phrase containing the answer
- British English throughout

OUTPUT FORMAT — follow exactly:

{{question_template}}
""".strip()

QUESTION_GENERATION_USER_ONE_TEMPLATE = """
Below is one example of a reading text paired with correctly written literal comprehension questions. Study how every answer is explicitly stated in the text and how the text reference quotes the supporting phrase.

{{exemplar_qa_mid}}

===

Now generate {{question_count}} NEW literal comprehension questions for the text below.
Do NOT copy the example questions. Write entirely new questions for the new text.

TEXT DETAILS:
- Grade: {{grade_label}}
- Topic: {{topic}}
- Text word count: {{word_count}}

THE TEXT:
Title: {{title}}

{{body}}

REQUIREMENTS:
- Exactly {{question_count}} questions
- ALL questions are literal — answer must be explicitly stated in the text
- No inference, no opinion, no general knowledge
- Questions follow the order of the text
- One unambiguous correct answer per question
- Three plausible but wrong distractors
- Text reference quotes the exact phrase containing the answer
- British English throughout

OUTPUT FORMAT — follow exactly:

{{question_template}}
""".strip()

QUESTION_GENERATION_USER_FEW_TEMPLATE = """
Below are three examples of a reading text paired with correctly written literal comprehension questions. Study them carefully. Notice how:
- Every answer is explicitly stated in the text — no inference needed
- The Text reference field quotes the exact phrase from the text containing the answer
- Questions follow the order of the text from beginning to end
- Wrong options are plausible but clearly wrong once the relevant sentence is found
- No question requires opinion, analysis, or general knowledge
- British English is used throughout

{{exemplar_qa_low}}

{{exemplar_qa_mid}}

{{exemplar_qa_high}}

===

Now generate {{question_count}} NEW literal comprehension questions for the text below.
Do NOT copy the example questions. Write entirely new questions for the new text.

TEXT DETAILS:
- Grade: {{grade_label}}
- Topic: {{topic}}
- Text word count: {{word_count}}

THE TEXT:
Title: {{title}}

{{body}}

REQUIREMENTS:
- Exactly {{question_count}} questions
- ALL questions are literal — answer must be explicitly stated in the text
- No inference, no opinion, no general knowledge
- Questions follow the order of the text
- One unambiguous correct answer per question
- Three plausible but wrong distractors
- Text reference quotes the exact phrase containing the answer
- British English throughout

OUTPUT FORMAT — follow exactly:

{{question_template}}
""".strip()

EVALUATION_SYSTEM_TEMPLATE = """
You are a careful educational evaluator for FrontRead.
You score generated reading materials for Scandinavian school students.
Use the rubric exactly. Be strict, especially when a response violates literal comprehension, British English, word count, or LIX constraints.
Return JSON only.
""".strip()

TEXT_EVALUATION_USER_TEMPLATE = """
Evaluate this generated FrontRead reading text on a 1-5 scale for each dimension.
5 = excellent / fully compliant. 4 = good. 3 = acceptable but needs review. 2 = weak. 1 = unacceptable.

TARGET PARAMETERS:
- Grade: {{grade_label}} ({{grade_key}})
- Topic: {{topic}}
- Text type: {{text_type_label}} / {{text_format}}
- Target word count: {{word_count}}
- Target LIX: {{lix_target}} with required band {{lix_min}}–{{lix_max}}

PYTHON-COMPUTED METRICS:
{{metrics_json}}

TEXT:
Title: {{title}}

{{body}}

DIMENSIONS:
{{dimensions}}

Respond only with JSON in exactly this structure:
{{json_schema}}
""".strip()

QUESTION_EVALUATION_USER_TEMPLATE = """
Evaluate this generated FrontRead literal comprehension question set on a 1-5 scale for each dimension.
5 = excellent / fully compliant. 4 = good. 3 = acceptable but needs review. 2 = weak. 1 = unacceptable.

TARGET PARAMETERS:
- Grade: {{grade_label}} ({{grade_key}})
- Topic: {{topic}}
- Target question count: {{question_count}}

PYTHON-COMPUTED QUESTION METRICS:
{{q_metrics_json}}

SOURCE TEXT:
Title: {{title}}

{{body}}

GENERATED QUESTIONS:
{{questions_json}}

DIMENSIONS:
{{dimensions}}

Respond only with JSON in exactly this structure:
{{json_schema}}
""".strip()


def normalise_strategy(strategy: str | None) -> str:
    key = (strategy or "few_shot").strip().lower().replace("-", "_")
    return key if key in PROMPT_STRATEGIES else "few_shot"


def build_question_template(question_count: int) -> str:
    blocks = []
    for i in range(1, int(question_count) + 1):
        blocks.append(
            f"""{i}. [Question]
A: [Option A]
B: [Option B]
C: [Option C]
D: [Option D]
Correct: [A/B/C/D]
Text reference: [Exact quote from the text]"""
        )
    return "\n\n".join(blocks)


def json_schema(dimensions: list[str]) -> str:
    body = {
        dim: {"score": "integer 1-5", "justification": "one concise sentence"}
        for dim in dimensions
    }
    return json.dumps(body, indent=2)


def render_prompt_template(template: str, context: dict[str, Any]) -> str:
    rendered = template or ""
    for key, value in context.items():
        rendered = rendered.replace("{{" + key + "}}", str(value))
    return rendered.strip()


def text_generation_user_template(strategy: str) -> str:
    strategy = normalise_strategy(strategy)
    if strategy == "zero_shot":
        return TEXT_GENERATION_USER_ZERO_TEMPLATE
    if strategy == "one_shot":
        return TEXT_GENERATION_USER_ONE_TEMPLATE
    return TEXT_GENERATION_USER_FEW_TEMPLATE


def question_generation_user_template(strategy: str) -> str:
    strategy = normalise_strategy(strategy)
    if strategy == "zero_shot":
        return QUESTION_GENERATION_USER_ZERO_TEMPLATE
    if strategy == "one_shot":
        return QUESTION_GENERATION_USER_ONE_TEMPLATE
    return QUESTION_GENERATION_USER_FEW_TEMPLATE


def default_prompt_overrides(text_strategy: str = "few_shot", question_strategy: str = "few_shot") -> dict[str, str]:
    return {
        "text_generation_system": TEXT_GENERATION_SYSTEM_TEMPLATE,
        "text_generation_user_template": text_generation_user_template(text_strategy),
        "question_generation_system": QUESTION_GENERATION_SYSTEM_TEMPLATE,
        "question_generation_user_template": question_generation_user_template(question_strategy),
        "text_evaluation_system": EVALUATION_SYSTEM_TEMPLATE,
        "text_evaluation_user_template": TEXT_EVALUATION_USER_TEMPLATE,
        "question_evaluation_system": EVALUATION_SYSTEM_TEMPLATE,
        "question_evaluation_user_template": QUESTION_EVALUATION_USER_TEMPLATE,
    }


def _template_from_overrides(overrides: dict[str, str] | None, key: str, default: str) -> str:
    if overrides and overrides.get(key):
        return overrides[key]
    return default


def generation_context(params: dict[str, Any]) -> dict[str, Any]:
    context = dict(params)
    context.update(
        {
            "exemplar_low": EXEMPLAR_TEXT_LOW,
            "exemplar_mid": EXEMPLAR_TEXT_MID,
            "exemplar_high": EXEMPLAR_TEXT_HIGH,
            "exemplar_qa_low": EXEMPLAR_QA_LOW,
            "exemplar_qa_mid": EXEMPLAR_QA_MID,
            "exemplar_qa_high": EXEMPLAR_QA_HIGH,
            "question_template": build_question_template(int(params.get("question_count", 4))),
        }
    )
    return context


def build_text_generation_prompts(
    params: dict[str, Any],
    strategy: str = "few_shot",
    prompt_overrides: dict[str, str] | None = None,
) -> tuple[str, str]:
    defaults = default_prompt_overrides(text_strategy=strategy, question_strategy=strategy)
    system_template = _template_from_overrides(
        prompt_overrides,
        "text_generation_system",
        defaults["text_generation_system"],
    )
    user_template = _template_from_overrides(
        prompt_overrides,
        "text_generation_user_template",
        defaults["text_generation_user_template"],
    )
    return system_template.strip(), render_prompt_template(user_template, generation_context(params))


def build_question_generation_prompts(
    params: dict[str, Any],
    title: str,
    body: str,
    strategy: str = "few_shot",
    prompt_overrides: dict[str, str] | None = None,
) -> tuple[str, str]:
    defaults = default_prompt_overrides(text_strategy=strategy, question_strategy=strategy)
    context = generation_context(params)
    context.update({"title": title, "body": body})
    system_template = _template_from_overrides(
        prompt_overrides,
        "question_generation_system",
        defaults["question_generation_system"],
    )
    user_template = _template_from_overrides(
        prompt_overrides,
        "question_generation_user_template",
        defaults["question_generation_user_template"],
    )
    return system_template.strip(), render_prompt_template(user_template, context)


def text_eval_dimensions_text() -> str:
    return "\n".join(
        [
            "- age_appropriateness: Is the text suitable for the target grade's age and reading ability?",
            "- topic_relevance: Does it stay focused on the requested topic?",
            "- engagement: Is it likely to be interesting and readable for students?",
            "- paragraph_consistency: Do all paragraphs maintain a similar difficulty level?",
            "- new_concepts_control: Are unfamiliar concepts limited and explained?",
            "- british_english: Does it use British English spelling and phrasing?",
            "- narrative_arc: Does it have a clear beginning, development, and ending, even for non-fiction?",
            "- vocabulary_fit: Is the vocabulary appropriate for this grade and LIX band?",
        ]
    )


def question_eval_dimensions_text() -> str:
    return "\n".join(
        [
            "- answerable_from_text: Can every answer be found explicitly in the passage?",
            "- single_correct_answer: Does every question have exactly one unambiguous correct option?",
            "- not_opinion_based: Are there no opinion/personal-response questions?",
            "- not_general_knowledge: Must students read the passage rather than rely on outside knowledge?",
            "- no_inference_required: Are all questions literal rather than inferential?",
            "- distractor_quality: Are wrong options plausible but clearly wrong from the text?",
            "- question_clarity: Is wording clear and grade-appropriate?",
            "- text_order: Do questions follow the order of the source passage?",
            "- text_reference_accuracy: Does each text reference contain the answer?",
            "- british_english: Do questions and options use British English?",
        ]
    )


def build_text_evaluation_prompts(
    title: str,
    body: str,
    params: dict[str, Any],
    metrics: dict[str, Any],
    prompt_overrides: dict[str, str] | None = None,
) -> tuple[str, str]:
    defaults = default_prompt_overrides()
    system_template = _template_from_overrides(
        prompt_overrides,
        "text_evaluation_system",
        defaults["text_evaluation_system"],
    )
    user_template = _template_from_overrides(
        prompt_overrides,
        "text_evaluation_user_template",
        defaults["text_evaluation_user_template"],
    )
    context = dict(params)
    context.update(
        {
            "title": title,
            "body": body,
            "metrics_json": json.dumps(metrics, ensure_ascii=False, indent=2, default=str),
            "dimensions": text_eval_dimensions_text(),
            "json_schema": json_schema(TEXT_EVAL_DIMS),
        }
    )
    return system_template.strip(), render_prompt_template(user_template, context)


def build_question_evaluation_prompts(
    title: str,
    body: str,
    questions: list[dict[str, Any]],
    params: dict[str, Any],
    q_metrics: dict[str, Any],
    prompt_overrides: dict[str, str] | None = None,
) -> tuple[str, str]:
    defaults = default_prompt_overrides()
    system_template = _template_from_overrides(
        prompt_overrides,
        "question_evaluation_system",
        defaults["question_evaluation_system"],
    )
    user_template = _template_from_overrides(
        prompt_overrides,
        "question_evaluation_user_template",
        defaults["question_evaluation_user_template"],
    )
    context = dict(params)
    context.update(
        {
            "title": title,
            "body": body,
            "questions_json": json.dumps(questions, ensure_ascii=False, indent=2),
            "q_metrics_json": json.dumps(q_metrics, ensure_ascii=False, indent=2, default=str),
            "dimensions": question_eval_dimensions_text(),
            "json_schema": json_schema(QUESTION_EVAL_DIMS),
        }
    )
    return system_template.strip(), render_prompt_template(user_template, context)
