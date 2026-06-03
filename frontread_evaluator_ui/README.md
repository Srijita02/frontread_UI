# FrontRead Prompt-Strategy Generator & Evaluator UI

A Streamlit prototype for FrontRead evaluators. It generates LIX-controlled reading passages from student/evaluator topics, creates literal comprehension questions from the final passage, evaluates both with deterministic Python metrics and AI rubrics, then flags outputs as approved or needing review.

The pipeline is fully decoupled. Evaluators can choose a different model for each stage:

1. Text generation
2. Question generation
3. Text AI evaluation
4. Question AI evaluation

This supports experiments such as GPT for text generation, Mistral/Ollama for question generation, Claude for text evaluation, and Gemini for question evaluation.

## What is included

- Clean Streamlit UI for evaluators
- Separate model/provider/API-key controls for all four stages
- One-click full pipeline, plus independent buttons for each stage
- Ability to load a previous `results.json` and continue from question generation or evaluation
- Topic input from manual entry, CSV upload, student-interest builder, or theme picker
- Multi-grade generation for one grade, several grades, or all Grades 1-10
- Zero-shot, one-shot, and few-shot prompting options for text generation
- Zero-shot, one-shot, and few-shot prompting options for question generation
- Sidebar button to view/edit prompt templates for text generation, question generation, text evaluation, and question evaluation
- Text metrics: word count, LIX, LIX band compliance, sentence count, average sentence length, long-word percentage, paragraph-level LIX
- Question metrics: target vs actual count, answer distribution, type distribution, four-option completeness, text-reference completeness
- AI text evaluation rubric with 1-5 scores
- AI question evaluation rubric with 1-5 scores
- Red/green approval flags based on LIX, word count, structural checks, and AI score thresholds
- Export to JSON, CSV, Excel, Markdown files, and a ZIP archive
- CLI commands for each separate stage

## Quick start

```bash
cd frontread_evaluator_ui
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

On Windows, if `python` is not available in PowerShell but Python works in VS Code, try:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

## API keys and local models

For OpenAI, Anthropic, or Google, paste the API key into the relevant stage panel in the sidebar. You can also set environment variables:

```bash
export OPENAI_API_KEY="your_key_here"
export ANTHROPIC_API_KEY="your_key_here"
export GOOGLE_API_KEY="your_key_here"
```

For local open-source models, run Ollama first:

```bash
ollama serve
ollama pull llama3.2
ollama pull mistral
```

Then select `Ollama / Llama 3.2 local` or `Ollama / Mistral local` for the desired stage.

## Test without an API key

Select `Demo backend / no API` for all stages. This generates placeholder content and fake AI scores so you can verify the UI, reports, stage buttons, and exports without spending API tokens.

## Prompt controls

In the sidebar, use **Prompting** to choose the prompting technique for text generation and question generation. The default is few-shot, but zero-shot and one-shot can be selected independently.

Click **View / edit prompts for 4 stages** to inspect or modify prompt templates for:

1. Text generation
2. Question generation
3. Text evaluation
4. Question evaluation

Edited templates are only used when **Use edited prompts when running** is enabled. Prompt templates use placeholders such as `{{topic}}`, `{{grade_label}}`, `{{body}}`, and `{{question_template}}`. Exports record the selected prompt strategies and whether custom prompts were used.

## UI workflow

### Full pipeline

1. Choose stage-specific models in the sidebar.
2. Enter one or more topics.
3. Choose target word count and question count.
4. Select one grade, multiple grades, or all grades.
5. Click **Run full pipeline**.
6. Review generated texts, questions, Python metrics, AI evaluations, and approval flags.
7. Export the full run ZIP, Excel report, CSV summary, or JSON.

### Separate stages

Use the buttons in order when you want more control:

1. **Generate texts**: creates only passages and Python text metrics.
2. **Generate questions**: uses the saved/generated texts and creates question sets.
3. **Evaluate texts**: runs only the text AI rubric.
4. **Evaluate questions**: runs only the question AI rubric.

You can load a previous `results.json` in the UI and continue from a later stage.

## Approval logic

A text is approved only when:

- Actual word count is within the configured tolerance, default +/-20 words.
- Actual LIX is within the selected grade band.
- AI qualitative text average score is at least the selected threshold, default 4.0.
- No individual AI text dimension is below the selected minimum, default 3.0.

A question set is approved only when:

- Generated question count matches the target.
- Every question has A-D options and one valid correct answer.
- AI qualitative question average score is at least the selected threshold.
- No individual AI question dimension is below the selected minimum.

If a stage has not been run yet, its approval flag remains red/false so evaluators know the item is incomplete.

## CLI usage

Generate text only:

```bash
python cli.py text \
  --provider openai \
  --model gpt-4o-mini \
  --topic "endangered animals" \
  --grades grade_3 grade_6 \
  --word-count 400 \
  --question-count 4 \
  --text-prompt-strategy few_shot \
  --out outputs/text_generation_only.json
```

Generate questions from a saved results JSON:

```bash
python cli.py questions \
  --provider openai \
  --model gpt-4o-mini \
  --results-json outputs/text_generation_only.json \
  --question-prompt-strategy few_shot \
  --out outputs/question_generation_only.json
```

Evaluate texts only:

```bash
python cli.py eval-text \
  --provider openai \
  --model gpt-4o-mini \
  --results-json outputs/question_generation_only.json \
  --out outputs/text_evaluation_only.json
```

Evaluate questions only:

```bash
python cli.py eval-questions \
  --provider openai \
  --model gpt-4o-mini \
  --results-json outputs/text_evaluation_only.json \
  --out outputs/question_evaluation_only.json
```

Run the full pipeline with different models per stage:

```bash
python cli.py full \
  --text-provider openai \
  --text-model gpt-4o-mini \
  --question-provider ollama \
  --question-model mistral \
  --text-eval-provider anthropic \
  --text-eval-model claude-3-5-haiku-latest \
  --question-eval-provider google \
  --question-eval-model gemini-2.0-flash \
  --text-prompt-strategy few_shot \
  --question-prompt-strategy few_shot \
  --topic "the water cycle" \
  --grades grade_3 grade_6 grade_9 \
  --word-count 400 \
  --question-count 4
```

For local demo testing:

```bash
python cli.py full \
  --text-provider demo --text-model demo \
  --question-provider demo --question-model demo \
  --text-eval-provider demo --text-eval-model demo \
  --question-eval-provider demo --question-eval-model demo \
  --text-prompt-strategy zero_shot \
  --question-prompt-strategy one_shot \
  --topic "volcanoes" \
  --grades grade_3
```

## Project structure

```text
frontread_evaluator_ui/
├── app.py
├── cli.py
├── requirements.txt
├── .env.example
├── sample_inputs/
│   └── student_topics.csv
├── frontread_core/
│   ├── config.py
│   ├── evaluation.py
│   ├── metrics.py
│   ├── model_client.py
│   ├── parsers.py
│   ├── pipeline.py
│   ├── prompt_templates.py
│   ├── prompts_fewshot.py
│   └── reporting.py
└── outputs/
    └── runs/
```

## Notes

- The UI does not permanently store API keys. They are used for the current Streamlit session only.
- Exported reports redact model API keys.
- The included few-shot prompt file fixes the high-level Barbie exemplar metadata to Grade 9 / LIX 33.5.
- Prompt strategy choices and custom-prompt usage are exported in JSON, CSV, and Excel reports.
- The app is designed as a prototype. Before production deployment, add authentication, database-backed run storage, stricter validation, and human revision submission workflows.
