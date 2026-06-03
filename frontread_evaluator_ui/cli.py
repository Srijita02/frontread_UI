from __future__ import annotations

import argparse
import json
from pathlib import Path

from frontread_core.config import GRADE_LEVELS
from frontread_core.model_client import ModelSpec
from frontread_core.prompt_templates import PROMPT_STRATEGIES
from frontread_core.pipeline import (
    build_params,
    generate_questions,
    generate_text,
    public_model_spec,
    run_pipeline,
    run_question_eval_stage,
    run_question_stage,
    run_text_eval_stage,
    run_text_stage,
)
from frontread_core.reporting import save_run

PROVIDERS = ["demo", "openai", "anthropic", "google", "ollama"]


def model_spec(args: argparse.Namespace, prefix: str = "") -> ModelSpec:
    p = f"{prefix}_" if prefix else ""
    provider = getattr(args, f"{p}provider")
    model = getattr(args, f"{p}model")
    api_key = getattr(args, f"{p}api_key", None)
    return ModelSpec(provider=provider, model=model, api_key=api_key, display_name=f"{provider}:{model}")


def read_results(path: str) -> list[dict]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    return [payload]


def write_json(path: str, payload: object) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"Saved to {out}")


def cmd_text(args: argparse.Namespace) -> None:
    spec = model_spec(args)
    results = []
    for grade_key in args.grades:
        params = build_params(
            grade_key=grade_key,
            topic=args.topic,
            word_count=args.word_count,
            question_count=args.question_count,
            text_type_key=args.text_type,
            text_format=args.text_format,
        )
        text_record = generate_text(
            spec,
            params,
            max_revisions=args.max_revisions,
            prompt_strategy=args.text_prompt_strategy,
        )
        results.append(
            {
                "topic": args.topic,
                "grade_key": grade_key,
                "grade_label": params["grade_label"],
                "models": {"text_generator": public_model_spec(spec)},
                "generator": public_model_spec(spec),
                "params": params,
                "prompting": {
                    "text_generation_strategy": args.text_prompt_strategy,
                    "question_generation_strategy": args.question_prompt_strategy,
                    "custom_prompts_used": False,
                },
                "text": text_record,
            }
        )
    write_json(args.out, results)


def cmd_questions(args: argparse.Namespace) -> None:
    spec = model_spec(args)
    results = read_results(args.results_json)
    results = run_question_stage(results, spec, prompt_strategy=args.question_prompt_strategy)
    write_json(args.out, results)


def cmd_eval_text(args: argparse.Namespace) -> None:
    spec = model_spec(args)
    results = read_results(args.results_json)
    results = run_text_eval_stage(results, spec)
    write_json(args.out, results)


def cmd_eval_questions(args: argparse.Namespace) -> None:
    spec = model_spec(args)
    results = read_results(args.results_json)
    results = run_question_eval_stage(results, spec)
    write_json(args.out, results)


def cmd_full(args: argparse.Namespace) -> None:
    text_generator = model_spec(args, "text")
    question_generator = model_spec(args, "question")
    text_evaluator = model_spec(args, "text_eval")
    question_evaluator = model_spec(args, "question_eval")
    results = run_pipeline(
        text_generator=text_generator,
        question_generator=question_generator,
        text_evaluator=text_evaluator,
        question_evaluator=question_evaluator,
        topics=[args.topic],
        grade_keys=args.grades,
        word_count=args.word_count,
        question_count=args.question_count,
        text_type_key=args.text_type,
        text_format=args.text_format,
        max_revisions=args.max_revisions,
        run_text_ai_eval=not args.skip_text_ai_eval,
        run_question_ai_eval=not args.skip_question_ai_eval,
        text_prompt_strategy=args.text_prompt_strategy,
        question_prompt_strategy=args.question_prompt_strategy,
    )
    paths = save_run(
        results,
        {
            "cli_args": vars(args),
            "models": {
                "text_generator": public_model_spec(text_generator),
                "question_generator": public_model_spec(question_generator),
                "text_evaluator": public_model_spec(text_evaluator),
                "question_evaluator": public_model_spec(question_evaluator),
            },
        },
    )
    print(f"Saved full run to {paths['run_dir']}")
    print(f"ZIP: {paths['zip']}")


def add_generation_settings(p: argparse.ArgumentParser) -> None:
    p.add_argument("--topic", default="medieval life and castles")
    p.add_argument("--grades", nargs="+", default=["grade_3"], choices=list(GRADE_LEVELS.keys()))
    p.add_argument("--word-count", type=int, default=400)
    p.add_argument("--question-count", type=int, default=4)
    p.add_argument("--text-type", default="mixed", choices=["mixed", "fiction", "non_fiction"])
    p.add_argument("--text-format", default=None)
    p.add_argument("--max-revisions", type=int, default=2)
    p.add_argument("--text-prompt-strategy", default="few_shot", choices=list(PROMPT_STRATEGIES.keys()))
    p.add_argument("--question-prompt-strategy", default="few_shot", choices=list(PROMPT_STRATEGIES.keys()))


def add_single_model(p: argparse.ArgumentParser) -> None:
    p.add_argument("--provider", default="demo", choices=PROVIDERS)
    p.add_argument("--model", default="demo")
    p.add_argument("--api-key", default=None)


def add_prefixed_model(p: argparse.ArgumentParser, prefix: str, default_provider: str = "demo", default_model: str = "demo") -> None:
    p.add_argument(f"--{prefix}-provider", default=default_provider, choices=PROVIDERS)
    p.add_argument(f"--{prefix}-model", default=default_model)
    p.add_argument(f"--{prefix}-api-key", default=None)


def main() -> None:
    parser = argparse.ArgumentParser(description="FrontRead few-shot generation CLI with decoupled stages")
    sub = parser.add_subparsers(dest="command", required=True)

    p_text = sub.add_parser("text", help="Generate texts only")
    add_single_model(p_text)
    add_generation_settings(p_text)
    p_text.add_argument("--out", default="outputs/text_generation_only.json")
    p_text.set_defaults(func=cmd_text)

    p_questions = sub.add_parser("questions", help="Generate questions from saved results JSON")
    add_single_model(p_questions)
    p_questions.add_argument("--results-json", required=True)
    p_questions.add_argument("--question-prompt-strategy", default="few_shot", choices=list(PROMPT_STRATEGIES.keys()))
    p_questions.add_argument("--out", default="outputs/question_generation_only.json")
    p_questions.set_defaults(func=cmd_questions)

    p_eval_text = sub.add_parser("eval-text", help="Run text AI evaluation on saved results JSON")
    add_single_model(p_eval_text)
    p_eval_text.add_argument("--results-json", required=True)
    p_eval_text.add_argument("--out", default="outputs/text_evaluation_only.json")
    p_eval_text.set_defaults(func=cmd_eval_text)

    p_eval_questions = sub.add_parser("eval-questions", help="Run question AI evaluation on saved results JSON")
    add_single_model(p_eval_questions)
    p_eval_questions.add_argument("--results-json", required=True)
    p_eval_questions.add_argument("--out", default="outputs/question_evaluation_only.json")
    p_eval_questions.set_defaults(func=cmd_eval_questions)

    p_full = sub.add_parser("full", help="Generate text, questions, and evaluations with independent models")
    add_generation_settings(p_full)
    add_prefixed_model(p_full, "text")
    add_prefixed_model(p_full, "question")
    add_prefixed_model(p_full, "text-eval")
    add_prefixed_model(p_full, "question-eval")
    p_full.add_argument("--skip-text-ai-eval", action="store_true")
    p_full.add_argument("--skip-question-ai-eval", action="store_true")
    p_full.set_defaults(func=cmd_full)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
