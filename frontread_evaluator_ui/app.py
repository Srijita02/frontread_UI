from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from frontread_core.config import (
    DEFAULT_APPROVAL_THRESHOLD,
    DEFAULT_MAX_REVISIONS,
    DEFAULT_MIN_DIMENSION_SCORE,
    DEFAULT_WORD_TOLERANCE,
    GRADE_LEVELS,
    MODEL_PRESETS,
    TEXT_TYPES,
    THEMES,
    WORD_COUNT_PRESETS,
)
from frontread_core.model_client import ModelSpec
from frontread_core.prompt_templates import PROMPT_STRATEGIES, default_prompt_overrides
from frontread_core.pipeline import (
    public_model_spec,
    run_pipeline,
    run_question_eval_stage,
    run_question_stage,
    run_text_eval_stage,
    run_text_stage,
)
from frontread_core.reporting import save_run, summary_rows

st.set_page_config(
    page_title="FrontRead Generator Evaluator",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-header {
        padding: 1.2rem 1.4rem;
        border-radius: 1.2rem;
        background: linear-gradient(135deg, #f7fbff 0%, #f4f7ff 48%, #fff8f3 100%);
        border: 1px solid #e6edf7;
        margin-bottom: 1rem;
    }
    .main-header h1 { margin-bottom: 0.2rem; }
    .subtitle { color: #5b6472; font-size: 1.02rem; }
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.65rem;
        border-radius: 999px;
        font-size: 0.82rem;
        font-weight: 700;
        margin-right: 0.35rem;
        margin-bottom: 0.25rem;
    }
    .approved { background: #e7f7ee; color: #137347; border: 1px solid #b8e5ca; }
    .review { background: #fff1f0; color: #b42318; border: 1px solid #ffd2cc; }
    .warn { background: #fff7e6; color: #946200; border: 1px solid #ffe1a6; }
    .metric-card {
        padding: 0.8rem;
        border-radius: 0.8rem;
        border: 1px solid #edf0f5;
        background: #ffffff;
        box-shadow: 0 1px 6px rgba(18, 38, 63, 0.04);
        min-height: 5.3rem;
    }
    .metric-title { color: #6b7280; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.04em; }
    .metric-value { font-size: 1.28rem; font-weight: 750; margin-top: 0.25rem; }
    .small-note { color: #667085; font-size: 0.88rem; }
    .stage-box {
        padding: 0.85rem 1rem;
        border-radius: 0.9rem;
        border: 1px solid #edf0f5;
        background: #fbfcff;
        margin-bottom: 0.6rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def grade_label(key: str) -> str:
    grade = GRADE_LEVELS[key]
    return f"{grade['label']} · LIX {grade['lix_range'][0]}–{grade['lix_range'][1]}"


def parse_topics(text: str) -> list[str]:
    if not text:
        return []
    parts: list[str] = []
    for line in text.splitlines():
        parts.extend(piece.strip() for piece in line.split(","))
    return [p for p in parts if p]


def get_default_question_count(word_count: int) -> int:
    if word_count in WORD_COUNT_PRESETS:
        return WORD_COUNT_PRESETS[word_count]["question_count"]
    return 4 if word_count <= 500 else 10


def build_model_spec(label: str, model_value: str, api_key: str | None) -> ModelSpec:
    preset = MODEL_PRESETS[label]
    model_id = model_value.strip() or preset["model"]
    return ModelSpec(
        provider=preset["provider"],
        model=model_id,
        api_key=(api_key or "").strip() or None,
        display_name=f"{label} · {model_id}",
    )


def model_picker(stage_label: str, key_prefix: str, default_index: int = 0) -> ModelSpec:
    label = st.selectbox(
        f"{stage_label} preset",
        list(MODEL_PRESETS.keys()),
        index=default_index,
        key=f"{key_prefix}_preset",
    )
    preset = MODEL_PRESETS[label]
    st.caption(preset["help"])
    model_id = st.text_input(
        f"{stage_label} model ID",
        value=preset["model"],
        key=f"{key_prefix}_model_id",
    )
    api_key = ""
    if preset["needs_key"]:
        api_key = st.text_input(
            f"{stage_label} API key",
            type="password",
            key=f"{key_prefix}_api_key",
            help=f"Uses {preset['key_env']} if left blank.",
        )
    return build_model_spec(label, model_id, api_key)


def status_badge(label: str, ok: bool | None) -> str:
    if ok is True:
        return f'<span class="status-badge approved">✓ {label}</span>'
    if ok is False:
        return f'<span class="status-badge review">✕ {label}</span>'
    return f'<span class="status-badge warn">– {label}</span>'


def metric_card(title: str, value: Any, note: str = "") -> None:
    display = "—" if value is None or value == "" else value
    st.markdown(
        f"""
        <div class="metric-card">
          <div class="metric-title">{title}</div>
          <div class="metric-value">{display}</div>
          <div class="small-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def ai_eval_dataframe(ai_eval: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for dim, payload in (ai_eval or {}).items():
        if dim.startswith("_") or not isinstance(payload, dict):
            continue
        rows.append(
            {
                "dimension": dim,
                "score": payload.get("score"),
                "justification": payload.get("justification"),
            }
        )
    return pd.DataFrame(rows)


def question_dataframe(questions: list[dict[str, Any]]) -> pd.DataFrame:
    cols = ["number", "question", "A", "B", "C", "D", "correct", "text_reference"]
    if not questions:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(questions)
    for col in cols:
        if col not in df.columns:
            df[col] = ""
    return df[cols]


def has_texts(results: list[dict[str, Any]]) -> bool:
    return any(r.get("text") and not r.get("error") for r in results)


def has_questions(results: list[dict[str, Any]]) -> bool:
    return any(r.get("questions") and not r.get("error") for r in results)


def run_config(stage: str, models: dict[str, ModelSpec], extra: dict[str, Any]) -> dict[str, Any]:
    return {
        "stage_completed": stage,
        "models": {name: public_model_spec(spec) for name, spec in models.items()},
        **extra,
    }


def persist_results(results: list[dict[str, Any]], config: dict[str, Any], persist: bool) -> dict[str, Path]:
    if not persist or not results:
        return {}
    return save_run(results, config)


st.markdown(
    """
    <div class="main-header">
      <h1>FrontRead Prompt-Strategy Generator & Evaluator</h1>
      <div class="subtitle">
        Choose models and zero-/one-/few-shot prompt strategies for generation, question writing, and evaluation review.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Stage-specific models")
    st.caption("Each step can use a different open-source or closed-source model.")

    with st.expander("1. Text generation model", expanded=True):
        text_generator = model_picker("Text generation", "text_gen", default_index=0)

    with st.expander("2. Question generation model", expanded=True):
        question_generator = model_picker("Question generation", "question_gen", default_index=0)

    with st.expander("3. Text evaluation model", expanded=True):
        text_evaluator = model_picker("Text evaluation", "text_eval", default_index=0)

    with st.expander("4. Question evaluation model", expanded=True):
        question_evaluator = model_picker("Question evaluation", "question_eval", default_index=0)

    st.divider()
    st.header("Prompting")
    st.caption("Choose prompting techniques for generation. Evaluation prompts are rubric-based but can also be viewed or edited.")

    strategy_keys = list(PROMPT_STRATEGIES.keys())
    text_prompt_strategy = st.selectbox(
        "Text generation prompting technique",
        strategy_keys,
        index=strategy_keys.index("few_shot"),
        format_func=lambda key: PROMPT_STRATEGIES[key],
        key="text_prompt_strategy",
    )
    question_prompt_strategy = st.selectbox(
        "Question generation prompting technique",
        strategy_keys,
        index=strategy_keys.index("few_shot"),
        format_func=lambda key: PROMPT_STRATEGIES[key],
        key="question_prompt_strategy",
    )

    if "show_prompt_panel" not in st.session_state:
        st.session_state["show_prompt_panel"] = False
    if st.button("View / edit prompts for 4 stages", width="stretch"):
        st.session_state["show_prompt_panel"] = not st.session_state["show_prompt_panel"]

    prompt_overrides = None
    custom_prompts_used = False
    prompt_defaults = default_prompt_overrides(text_prompt_strategy, question_prompt_strategy)

    if st.session_state["show_prompt_panel"]:
        st.caption("Prompt templates use placeholders such as {{topic}}, {{grade_label}}, {{body}}, and {{question_template}}.")
        use_edited_prompts = st.checkbox("Use edited prompts when running", value=False)
        custom_prompts_used = bool(use_edited_prompts)

        if st.button("Restore default prompt templates", width="stretch"):
            for key in list(st.session_state.keys()):
                if key.startswith("prompt_tpl_"):
                    del st.session_state[key]
            st.rerun()

        with st.expander("1. Text generation prompt", expanded=False):
            text_generation_system = st.text_area(
                "Text generation system prompt",
                value=prompt_defaults["text_generation_system"],
                height=140,
                key=f"prompt_tpl_text_gen_system_{text_prompt_strategy}",
            )
            text_generation_user_template = st.text_area(
                "Text generation user prompt template",
                value=prompt_defaults["text_generation_user_template"],
                height=320,
                key=f"prompt_tpl_text_gen_user_{text_prompt_strategy}",
            )

        with st.expander("2. Question generation prompt", expanded=False):
            question_generation_system = st.text_area(
                "Question generation system prompt",
                value=prompt_defaults["question_generation_system"],
                height=140,
                key=f"prompt_tpl_question_gen_system_{question_prompt_strategy}",
            )
            question_generation_user_template = st.text_area(
                "Question generation user prompt template",
                value=prompt_defaults["question_generation_user_template"],
                height=320,
                key=f"prompt_tpl_question_gen_user_{question_prompt_strategy}",
            )

        with st.expander("3. Text evaluation prompt", expanded=False):
            text_evaluation_system = st.text_area(
                "Text evaluation system prompt",
                value=prompt_defaults["text_evaluation_system"],
                height=120,
                key="prompt_tpl_text_eval_system",
            )
            text_evaluation_user_template = st.text_area(
                "Text evaluation user prompt template",
                value=prompt_defaults["text_evaluation_user_template"],
                height=300,
                key="prompt_tpl_text_eval_user",
            )

        with st.expander("4. Question evaluation prompt", expanded=False):
            question_evaluation_system = st.text_area(
                "Question evaluation system prompt",
                value=prompt_defaults["question_evaluation_system"],
                height=120,
                key="prompt_tpl_question_eval_system",
            )
            question_evaluation_user_template = st.text_area(
                "Question evaluation user prompt template",
                value=prompt_defaults["question_evaluation_user_template"],
                height=300,
                key="prompt_tpl_question_eval_user",
            )

        if use_edited_prompts:
            prompt_overrides = {
                "text_generation_system": text_generation_system,
                "text_generation_user_template": text_generation_user_template,
                "question_generation_system": question_generation_system,
                "question_generation_user_template": question_generation_user_template,
                "text_evaluation_system": text_evaluation_system,
                "text_evaluation_user_template": text_evaluation_user_template,
                "question_evaluation_system": question_evaluation_system,
                "question_evaluation_user_template": question_evaluation_user_template,
            }


    st.divider()
    st.header("Approval rules")
    approval_threshold = st.slider("Minimum AI average score", 1.0, 5.0, DEFAULT_APPROVAL_THRESHOLD, 0.1)
    min_dimension_score = st.slider("Minimum score per AI dimension", 1.0, 5.0, DEFAULT_MIN_DIMENSION_SCORE, 0.1)
    word_tolerance = st.number_input("Word-count tolerance", min_value=0, max_value=100, value=DEFAULT_WORD_TOLERANCE, step=5)
    max_revisions = st.slider("Max LIX/word-count revisions", 0, 4, DEFAULT_MAX_REVISIONS)
    run_text_ai_eval = st.checkbox("Run text AI evaluation in full pipeline", value=True)
    run_question_ai_eval = st.checkbox("Run question AI evaluation in full pipeline", value=True)
    temperature = st.slider("Generation temperature", 0.0, 1.0, 0.3, 0.05)
    persist_after_each_stage = st.checkbox("Save/export after each stage", value=True)

left, right = st.columns([1.2, 0.8], gap="large")

with left:
    st.subheader("1. Topic source")
    topic_mode = st.radio(
        "How should topics enter the pipeline?",
        ["Manual evaluator input", "Student topic CSV", "Student interest builder", "Theme/subtopic picker"],
        horizontal=True,
    )

    topics: list[str] = []
    if topic_mode == "Manual evaluator input":
        topic_text = st.text_area(
            "Topic(s)",
            value="medieval life and castles",
            help="Enter one topic per line, or separate topics with commas.",
            height=96,
        )
        topics = parse_topics(topic_text)
    elif topic_mode == "Student topic CSV":
        uploaded = st.file_uploader("Upload CSV with a 'topic' column", type=["csv"])
        if uploaded is not None:
            df_topics = pd.read_csv(uploaded)
            topic_col = "topic" if "topic" in df_topics.columns else df_topics.columns[0]
            topics = [str(x).strip() for x in df_topics[topic_col].dropna().tolist() if str(x).strip()]
            st.dataframe(df_topics.head(20), width="stretch")
        else:
            st.info("Upload a CSV from a student-interest form. The first column will be used if no 'topic' column exists.")
    elif topic_mode == "Student interest builder":
        c1, c2 = st.columns(2)
        with c1:
            student_name = st.text_input("Student name or ID", value="student_001")
            favourite = st.text_input("Student interest", value="space and robots")
        with c2:
            theme = st.selectbox("Theme", list(THEMES.keys()), index=1)
            challenge = st.selectbox("Preferred angle", ["adventure", "facts", "mystery", "daily life", "future"], index=1)
        topics = [f"{favourite} through a {challenge} lens for {student_name}"] if favourite else []
        st.caption("This prototype keeps the topic non-sensitive and does not store student identifiers outside exported run files.")
    else:
        theme = st.selectbox("Theme", list(THEMES.keys()), index=0)
        selected_subtopics = st.multiselect("Subtopics", THEMES[theme], default=THEMES[theme][:1])
        topics = selected_subtopics

with right:
    st.subheader("2. Generation settings")
    word_count = st.number_input("Target word count", min_value=100, max_value=1500, value=400, step=50)
    suggested_q = get_default_question_count(int(word_count))
    question_count = st.number_input("Desired number of questions", min_value=1, max_value=20, value=suggested_q, step=1)

    text_type_key = st.selectbox(
        "Text type",
        list(TEXT_TYPES.keys()),
        format_func=lambda key: TEXT_TYPES[key]["label"],
        index=2,
    )
    format_options = ["Auto"] + TEXT_TYPES[text_type_key]["formats"]
    text_format_choice = st.selectbox("Text format", format_options, index=0)
    text_format = None if text_format_choice == "Auto" else text_format_choice

    all_grades = st.checkbox("Generate for all grades / LIX levels", value=False)
    if all_grades:
        grade_keys = list(GRADE_LEVELS.keys())
        st.caption("Selected: Grades 1-10")
    else:
        grade_keys = st.multiselect(
            "Grades / LIX bands",
            list(GRADE_LEVELS.keys()),
            default=["grade_3", "grade_6", "grade_9"],
            format_func=grade_label,
        )

st.divider()
st.subheader("3. Workflow")
st.markdown(
    """
    <div class="stage-box">
    Run the whole pipeline in one click, or run each stage separately. This lets you generate texts with one model, generate questions with another, and evaluate text/question quality with different evaluator models.
    </div>
    """,
    unsafe_allow_html=True,
)

with st.expander("Load previous run JSON for separate question/evaluation stages", expanded=False):
    uploaded_json = st.file_uploader("Upload a previous results.json", type=["json"])
    if uploaded_json is not None:
        try:
            loaded = json.loads(uploaded_json.getvalue().decode("utf-8"))
            if isinstance(loaded, list):
                st.session_state["results"] = loaded
                st.session_state["paths"] = {}
                st.success(f"Loaded {len(loaded)} result item(s).")
            else:
                st.error("The uploaded JSON should contain a list of result objects.")
        except Exception as exc:
            st.error(f"Could not read JSON: {exc}")

models = {
    "text_generator": text_generator,
    "question_generator": question_generator,
    "text_evaluator": text_evaluator,
    "question_evaluator": question_evaluator,
}

common_config = {
    "topics": topics,
    "grades": grade_keys,
    "word_count": int(word_count),
    "question_count": int(question_count),
    "text_type": text_type_key,
    "text_format": text_format,
    "approval_threshold": approval_threshold,
    "min_dimension_score": min_dimension_score,
    "word_tolerance": word_tolerance,
    "max_revisions": max_revisions,
    "run_text_ai_eval": run_text_ai_eval,
    "run_question_ai_eval": run_question_ai_eval,
    "text_prompt_strategy": text_prompt_strategy,
    "question_prompt_strategy": question_prompt_strategy,
    "custom_prompts_used": custom_prompts_used,
    "prompt_overrides": prompt_overrides if custom_prompts_used else None,
}

btn_cols = st.columns([1, 1, 1, 1, 1.15, 0.75])
with btn_cols[0]:
    generate_texts_clicked = st.button("1. Generate texts", width="stretch")
with btn_cols[1]:
    generate_questions_clicked = st.button("2. Generate questions", width="stretch")
with btn_cols[2]:
    eval_texts_clicked = st.button("3. Evaluate texts", width="stretch")
with btn_cols[3]:
    eval_questions_clicked = st.button("4. Evaluate questions", width="stretch")
with btn_cols[4]:
    full_clicked = st.button("Run full pipeline", type="primary", width="stretch")
with btn_cols[5]:
    clear_clicked = st.button("Clear", width="stretch")

if clear_clicked:
    st.session_state.pop("results", None)
    st.session_state.pop("paths", None)
    st.success("Cleared current session results.")

progress = st.progress(0)
status = st.empty()


def progress_callback(stage: str, message: str, idx: int, total: int) -> None:
    progress.progress(idx / max(total, 1))
    status.info(f"{stage.replace('_', ' ').title()}: {message} ({idx}/{total})")


def need_topics_and_grades() -> bool:
    if not topics:
        st.error("Please provide at least one topic.")
        return False
    if not grade_keys:
        st.error("Please select at least one grade/LIX band.")
        return False
    return True

if generate_texts_clicked and need_topics_and_grades():
    with st.spinner("Running text generation stage..."):
        results = run_text_stage(
            text_generator=text_generator,
            topics=topics,
            grade_keys=grade_keys,
            word_count=int(word_count),
            question_count=int(question_count),
            text_type_key=text_type_key,
            text_format=text_format,
            approval_threshold=float(approval_threshold),
            min_dimension_score=float(min_dimension_score),
            word_tolerance=int(word_tolerance),
            max_revisions=int(max_revisions),
            temperature=float(temperature),
            prompt_strategy=text_prompt_strategy,
            prompt_overrides=prompt_overrides,
            progress_callback=progress_callback,
        )
        paths = persist_results(results, run_config("text_generation", models, common_config), persist_after_each_stage)
    st.session_state["results"] = results
    st.session_state["paths"] = paths
    status.success("Text generation stage complete.")

if generate_questions_clicked:
    current = st.session_state.get("results", [])
    if not has_texts(current):
        st.error("Generate or load texts before generating questions.")
    else:
        with st.spinner("Running question generation stage..."):
            results = run_question_stage(
                results=current,
                question_generator=question_generator,
                approval_threshold=float(approval_threshold),
                min_dimension_score=float(min_dimension_score),
                temperature=min(float(temperature), 0.3),
                prompt_strategy=question_prompt_strategy,
                prompt_overrides=prompt_overrides,
                progress_callback=progress_callback,
            )
            paths = persist_results(results, run_config("question_generation", models, common_config), persist_after_each_stage)
        st.session_state["results"] = results
        st.session_state["paths"] = paths
        status.success("Question generation stage complete.")

if eval_texts_clicked:
    current = st.session_state.get("results", [])
    if not has_texts(current):
        st.error("Generate or load texts before evaluating texts.")
    else:
        with st.spinner("Running text evaluation stage..."):
            results = run_text_eval_stage(
                results=current,
                text_evaluator=text_evaluator,
                approval_threshold=float(approval_threshold),
                min_dimension_score=float(min_dimension_score),
                prompt_overrides=prompt_overrides,
                progress_callback=progress_callback,
            )
            paths = persist_results(results, run_config("text_evaluation", models, common_config), persist_after_each_stage)
        st.session_state["results"] = results
        st.session_state["paths"] = paths
        status.success("Text evaluation stage complete.")

if eval_questions_clicked:
    current = st.session_state.get("results", [])
    if not has_questions(current):
        st.error("Generate or load question sets before evaluating questions.")
    else:
        with st.spinner("Running question evaluation stage..."):
            results = run_question_eval_stage(
                results=current,
                question_evaluator=question_evaluator,
                approval_threshold=float(approval_threshold),
                min_dimension_score=float(min_dimension_score),
                prompt_overrides=prompt_overrides,
                progress_callback=progress_callback,
            )
            paths = persist_results(results, run_config("question_evaluation", models, common_config), persist_after_each_stage)
        st.session_state["results"] = results
        st.session_state["paths"] = paths
        status.success("Question evaluation stage complete.")

if full_clicked and need_topics_and_grades():
    with st.spinner("Running full pipeline..."):
        results = run_pipeline(
            text_generator=text_generator,
            question_generator=question_generator,
            text_evaluator=text_evaluator,
            question_evaluator=question_evaluator,
            topics=topics,
            grade_keys=grade_keys,
            word_count=int(word_count),
            question_count=int(question_count),
            text_type_key=text_type_key,
            text_format=text_format,
            approval_threshold=float(approval_threshold),
            min_dimension_score=float(min_dimension_score),
            word_tolerance=int(word_tolerance),
            max_revisions=int(max_revisions),
            run_text_ai_eval=run_text_ai_eval,
            run_question_ai_eval=run_question_ai_eval,
            temperature=float(temperature),
            text_prompt_strategy=text_prompt_strategy,
            question_prompt_strategy=question_prompt_strategy,
            prompt_overrides=prompt_overrides,
            progress_callback=progress_callback,
        )
        paths = persist_results(results, run_config("full_pipeline", models, common_config), True)
    st.session_state["results"] = results
    st.session_state["paths"] = paths
    status.success("Full pipeline complete. Results saved and ready for review.")

results = st.session_state.get("results", [])
paths = st.session_state.get("paths", {})

if results:
    st.subheader("4. Review dashboard")
    summary_df = pd.DataFrame(summary_rows(results))
    c1, c2, c3, c4, c5 = st.columns(5)
    approved_total = int(summary_df.get("overall_approved", pd.Series(dtype=bool)).fillna(False).sum()) if not summary_df.empty else 0
    needs_review_total = len(summary_df) - approved_total
    with c1:
        metric_card("Generated items", len(summary_df), "topic x grade outputs")
    with c2:
        metric_card("Fully approved", approved_total, "green outputs")
    with c3:
        metric_card("Needs review", needs_review_total, "red outputs")
    with c4:
        avg_lix_diff = round(summary_df["lix_diff"].dropna().abs().mean(), 2) if "lix_diff" in summary_df else "—"
        metric_card("Avg |LIX diff|", avg_lix_diff, "smaller is better")
    with c5:
        q_done = int(summary_df.get("question_count_actual", pd.Series(dtype=float)).fillna(0).gt(0).sum()) if not summary_df.empty else 0
        metric_card("Question sets", q_done, "generated so far")

    filter_mode = st.radio("Filter results", ["Needs review first", "All", "Approved only", "Needs review only"], horizontal=True)
    display_df = summary_df.copy()
    if filter_mode == "Approved only" and "overall_approved" in display_df:
        display_df = display_df[display_df["overall_approved"] == True]
    elif filter_mode == "Needs review only" and "overall_approved" in display_df:
        display_df = display_df[display_df["overall_approved"] != True]
    elif filter_mode == "Needs review first" and "overall_approved" in display_df:
        display_df = display_df.sort_values(["overall_approved", "grade_key"], ascending=[True, True])

    st.dataframe(display_df, width="stretch", hide_index=True)

    if paths:
        st.markdown("#### Export latest saved stage")
        d1, d2, d3, d4 = st.columns(4)
        with d1:
            with open(paths["zip"], "rb") as f:
                st.download_button("Download full ZIP", f, file_name=Path(paths["zip"]).name, mime="application/zip", width="stretch")
        with d2:
            with open(paths["xlsx"], "rb") as f:
                st.download_button("Download Excel", f, file_name=Path(paths["xlsx"]).name, mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", width="stretch")
        with d3:
            with open(paths["csv"], "rb") as f:
                st.download_button("Download CSV", f, file_name=Path(paths["csv"]).name, mime="text/csv", width="stretch")
        with d4:
            with open(paths["json"], "rb") as f:
                st.download_button("Download JSON", f, file_name=Path(paths["json"]).name, mime="application/json", width="stretch")

    st.markdown("#### Generated texts, questions, and evaluations")
    ordered_results = results
    if filter_mode == "Needs review first":
        ordered_results = sorted(results, key=lambda r: (r.get("approval", {}).get("overall_approved") is True, r.get("grade_key", "")))
    elif filter_mode == "Approved only":
        ordered_results = [r for r in results if r.get("approval", {}).get("overall_approved") is True]
    elif filter_mode == "Needs review only":
        ordered_results = [r for r in results if r.get("approval", {}).get("overall_approved") is not True]

    for result in ordered_results:
        approval = result.get("approval", {})
        title = result.get("text", {}).get("title", "Untitled")
        expander_title = f"{result.get('grade_label')} · {result.get('topic')} · {title}"
        with st.expander(expander_title, expanded=approval.get("overall_approved") is not True):
            if result.get("error"):
                st.error(result["error"])
            if result.get("question_error"):
                st.warning(result["question_error"])

            st.markdown(
                status_badge("Text approved", approval.get("text_approved"))
                + status_badge("Questions approved", approval.get("questions_approved"))
                + status_badge("Overall approved", approval.get("overall_approved")),
                unsafe_allow_html=True,
            )

            models_used = result.get("models", {})
            st.caption(
                " | ".join(
                    [
                        f"Text gen: {models_used.get('text_generator', {}).get('display_name', 'not set')}",
                        f"Question gen: {models_used.get('question_generator', {}).get('display_name', 'not run')}",
                        f"Text eval: {models_used.get('text_evaluator', {}).get('display_name', 'not run')}",
                        f"Question eval: {models_used.get('question_evaluator', {}).get('display_name', 'not run')}",
                    ]
                )
            )
            prompting_used = result.get("prompting", {})
            st.caption(
                "Prompting: "
                + f"Text {PROMPT_STRATEGIES.get(prompting_used.get('text_generation_strategy'), prompting_used.get('text_generation_strategy', 'few_shot'))}"
                + " | "
                + f"Questions {PROMPT_STRATEGIES.get(prompting_used.get('question_generation_strategy'), prompting_used.get('question_generation_strategy', 'few_shot'))}"
                + (" | Custom prompt templates" if prompting_used.get("custom_prompts_used") else "")
            )

            text_record = result.get("text", {})
            q_record = result.get("questions", {})
            tm = text_record.get("metrics", {})
            qm = q_record.get("metrics", {})
            m1, m2, m3, m4, m5 = st.columns(5)
            with m1:
                metric_card("Words", tm.get("actual_word_count"), f"target {tm.get('target_word_count')} ±{tm.get('word_count_tolerance')}")
            with m2:
                metric_card("LIX", tm.get("lix_actual"), f"band {tm.get('lix_min')}–{tm.get('lix_max')}")
            with m3:
                metric_card("Text AI", approval.get("text_ai_overall_score"), f"threshold {approval_threshold}")
            with m4:
                metric_card("Questions", qm.get("question_count_actual"), f"target {qm.get('question_count_target')}")
            with m5:
                metric_card("Question AI", approval.get("question_ai_overall_score"), f"threshold {approval_threshold}")

            tab_text, tab_questions, tab_text_eval, tab_question_eval, tab_raw = st.tabs(
                ["Generated text", "Questions", "Text eval", "Question eval", "Raw metadata"]
            )
            with tab_text:
                st.text_area("Passage", value=text_record.get("body", ""), height=360, key=f"body_{result.get('run_item_id')}")
                if text_record:
                    st.caption(f"Revision iterations used: {text_record.get('final_revision')} · elapsed {text_record.get('elapsed_sec')}s")
            with tab_questions:
                st.dataframe(question_dataframe(q_record.get("questions", [])), width="stretch", hide_index=True)
                if qm:
                    st.caption(f"Answer distribution: {qm.get('answer_distribution')} · Type distribution: {qm.get('type_distribution')}")
            with tab_text_eval:
                st.json(tm, expanded=False)
                st.dataframe(ai_eval_dataframe(result.get("text_ai_eval", {})), width="stretch", hide_index=True)
            with tab_question_eval:
                st.json(qm, expanded=False)
                st.dataframe(ai_eval_dataframe(result.get("question_ai_eval", {})), width="stretch", hide_index=True)
            with tab_raw:
                st.json(result, expanded=False)
else:
    st.info("Configure the run, then run one stage or the full pipeline. Use the demo backend to test the UI without API keys.")
