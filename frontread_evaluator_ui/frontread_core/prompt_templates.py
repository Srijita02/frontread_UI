"""Prompt strategy templates and rendering helpers for the FrontRead UI.

The UI can expose these templates for viewing/editing. The templates use a
small Jinja-style placeholder syntax, for example {{grade_label}} or {{body}}.
This avoids Python .format() collisions with JSON snippets or rubric braces.
"""

from __future__ import annotations

import json
from typing import Any

from .prompts_fewshot import (
    EXEMPLAR_QA_HIGH,
    EXEMPLAR_QA_LOW,
    EXEMPLAR_QA_MID,
    EXEMPLAR_TEXT_HIGH,
    EXEMPLAR_TEXT_LOW,
    EXEMPLAR_TEXT_MID,
    TEXT_REVISION_PROMPT,
)

TEXT_EVAL_DIMS = [
    "british_english",
    "age_appropriateness",
    "topic_relevance",
    "engagement",
    "paragraph_consistency",
    "new_concepts_introduced",
    "narrative_arc",
    "vocabulary_fit",
    "overall_quality",
]

QUESTION_EVAL_DIMS = [
    "british_english",
    "all_answers_from_text",
    "single_correct_answer",
    "question_clarity",
    "text_order_maintained",
    "text_reference_quality",
    "not_opinion_based",
    "not_general_knowledge_based",
    "no_inference_needed",
    "distractor_quality",
    "overall_quality",
]

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
Evaluate this generated FrontRead reading text.

You are performing QUALITATIVE TEXT REVIEW ONLY.
Do NOT recalculate word count, LIX, sentence length, paragraph LIX, or any metric already provided.
Use the Python-computed metrics as factual inputs when judging readability, LIX compliance, paragraph consistency, and vocabulary fit.

SCORING:
1 = Very poor / unacceptable
2 = Weak
3 = Acceptable but needs review
4 = Good
5 = Excellent / fully compliant

TARGET PARAMETERS:
- Grade: {{grade_label}} ({{grade_key}})
- Topic: {{topic}}
- Text type: {{text_type_label}}
- Text format: {{text_format}}
- Target word count: {{word_count}}
- Target LIX: {{lix_target}}
- Required LIX band: {{lix_min}}-{{lix_max}}

PYTHON-COMPUTED METRICS:
{{metrics_json}}

TEXT:
Title: {{title}}

{{body}}

QUALITATIVE TEXT EVALUATION CRITERIA:
{{dimensions}}

IMPORTANT:
- Evaluate the text only. Do not evaluate questions.
- Use Python metrics as evidence, not as something to recompute.
- If actual LIX is outside the required band, reflect this in vocabulary_fit and age_appropriateness where relevant.
- If paragraph LIX values or paragraph quality are uneven, reflect this in paragraph_consistency.
- British English means UK spelling, grammar, punctuation conventions, and vocabulary.
- Return JSON only. Do not include markdown, commentary, or extra text.

Respond only with JSON in exactly this structure:
{{json_schema}}
""".strip()

QUESTION_EVALUATION_USER_TEMPLATE = """
Evaluate this generated FrontRead literal comprehension question set.

You are performing QUALITATIVE QUESTION REVIEW ONLY.
Do NOT recalculate question metrics already provided.
Use the Python-computed question metrics as factual inputs when judging completeness, answer distribution, references, and formatting.

SCORING:
1 = Very poor / unacceptable
2 = Weak
3 = Acceptable but needs review
4 = Good
5 = Excellent / fully compliant

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

QUALITATIVE QUESTION EVALUATION CRITERIA:
{{dimensions}}

IMPORTANT:
- Evaluate the questions only. Do not evaluate the source text quality.
- Use Python metrics as evidence, not as something to recompute.
- Penalise any question whose answer is not explicitly stated in the text.
- Penalise any question that requires inference, interpretation, opinion, or general knowledge.
- Penalise inaccurate or imprecise text references.
- Penalise questions with more than one plausible correct answer.
- British English means UK spelling, grammar, punctuation conventions, and vocabulary.
- Return JSON only. Do not include markdown, commentary, or extra text.

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
            "- british_english: Checks whether British English spelling, grammar, punctuation conventions, and vocabulary are used consistently. 1 = Mostly incorrect or inconsistent British English; 5 = fully consistent British English.",
            "- age_appropriateness: Measures whether sentence structure, ideas, and vocabulary suit the target grade level. 1 = completely unsuitable for the target age; 5 = perfectly matched to the target age.",
            "- topic_relevance: Evaluates how closely the text stays focused on the assigned topic. 1 = frequently off-topic; 5 = fully relevant throughout.",
            "- engagement: Assesses how interesting, lively, and engaging the text is for student readers. 1 = very dull or unengaging; 5 = highly engaging and readable.",
            "- paragraph_consistency: Checks whether all paragraphs maintain a similar complexity, tone, and structure. 1 = highly uneven paragraph quality; 5 = strong consistency throughout.",
            "- new_concepts_introduced: Evaluates whether the text limits unfamiliar concepts appropriately for reading practice. 1 = too many unfamiliar concepts introduced; 5 = appropriate and controlled concept load.",
            "- narrative_arc: Measures whether the text has a clear progression, flow, and satisfying structure. 1 = no clear structure or progression; 5 = strong and coherent narrative flow.",
            "- vocabulary_fit: Assesses whether the vocabulary matches the target reading level and remains consistent. 1 = vocabulary unsuitable or inconsistent; 5 = excellent vocabulary alignment.",
            "- overall_quality: Overall holistic quality score across all text criteria. 1 = very weak overall quality; 5 = excellent overall quality.",
        ]
    )


def question_eval_dimensions_text() -> str:
    return "\n".join(
        [
            "- british_english: Checks whether the questions and options consistently use British English spelling, vocabulary, and style. 1 = mostly American English or inconsistent usage; 5 = fully consistent British English throughout.",
            "- all_answers_from_text: Verifies that every correct answer is stated directly and explicitly in the text. 1 = many answers not found in text; 5 = every answer explicitly stated in text.",
            "- single_correct_answer: Ensures each question has exactly one unambiguous correct answer. 1 = multiple possible answers in several questions; 5 = only one clearly correct answer per question.",
            "- question_clarity: Measures how clearly and simply the questions are written for students. 1 = confusing or unclear wording; 5 = very clear and easy to understand.",
            "- text_order_maintained: Checks whether questions follow the sequence of information in the text. 1 = questions appear in random order; 5 = perfectly follows text order.",
            "- text_reference_quality: Evaluates whether the text references accurately quote the exact supporting phrase or sentence. 1 = missing or inaccurate references; 5 = precise and directly relevant references.",
            "- not_opinion_based: Confirms that questions ask only factual information, not opinions or interpretations. 1 = mostly opinion-based; 5 = entirely factual questions.",
            "- not_general_knowledge_based: Ensures questions cannot be answered using general knowledge alone without reading the text. 1 = easily answerable from general knowledge; 5 = requires reading the text.",
            "- no_inference_needed: Checks that students do not need to infer, analyse, or interpret beyond explicit information. 1 = heavy inference required; 5 = completely literal comprehension only.",
            "- distractor_quality: Evaluates whether incorrect options are plausible but clearly wrong according to the text. 1 = silly or obvious distractors; 5 = plausible and well-balanced distractors.",
            "- overall_quality: Overall averaged quality rating across all question evaluation criteria. 1 = very weak overall quality; 5 = excellent overall quality.",
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
