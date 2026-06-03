"""Central configuration for the FrontRead evaluator UI."""

from __future__ import annotations

GRADE_LEVELS = {
    "grade_1": {
        "label": "1st Grade",
        "age_range": [6, 7],
        "lix_range": [0, 14],
        "lix_target": 10,
        "vocabulary_notes": "Very simple, high-frequency words only. One idea per sentence. Short sentences.",
        "sentence_length_avg": 6,
    },
    "grade_2": {
        "label": "2nd Grade",
        "age_range": [7, 8],
        "lix_range": [10, 19],
        "lix_target": 15,
        "vocabulary_notes": "Simple familiar words. Sentences can be slightly longer. Concrete ideas only.",
        "sentence_length_avg": 8,
    },
    "grade_3": {
        "label": "3rd Grade",
        "age_range": [8, 9],
        "lix_range": [15, 24],
        "lix_target": 20,
        "vocabulary_notes": "Everyday vocabulary. Occasional new word explained immediately in context.",
        "sentence_length_avg": 10,
    },
    "grade_4": {
        "label": "4th Grade",
        "age_range": [9, 10],
        "lix_range": [20, 29],
        "lix_target": 25,
        "vocabulary_notes": "Broader vocabulary. Some compound sentences. At most one new term per text.",
        "sentence_length_avg": 12,
    },
    "grade_5": {
        "label": "5th Grade",
        "age_range": [10, 11],
        "lix_range": [25, 34],
        "lix_target": 30,
        "vocabulary_notes": "Varied vocabulary. Mix of simple and compound sentences.",
        "sentence_length_avg": 13,
    },
    "grade_6": {
        "label": "6th Grade",
        "age_range": [11, 12],
        "lix_range": [30, 39],
        "lix_target": 34,
        "vocabulary_notes": "Moderate complexity. Compound and complex sentences. Some topic-specific vocabulary.",
        "sentence_length_avg": 15,
    },
    "grade_7": {
        "label": "7th Grade",
        "age_range": [12, 13],
        "lix_range": [34, 42],
        "lix_target": 38,
        "vocabulary_notes": "Richer vocabulary. Clear topic sentences per paragraph. Domain words used in context.",
        "sentence_length_avg": 16,
    },
    "grade_8": {
        "label": "8th Grade",
        "age_range": [13, 14],
        "lix_range": [38, 46],
        "lix_target": 42,
        "vocabulary_notes": "Rich vocabulary. Complex sentence structures acceptable. Precise word choice.",
        "sentence_length_avg": 17,
    },
    "grade_9": {
        "label": "9th Grade",
        "age_range": [14, 15],
        "lix_range": [42, 50],
        "lix_target": 46,
        "vocabulary_notes": "Academic vocabulary. Dense paragraphs. Formal register where appropriate.",
        "sentence_length_avg": 18,
    },
    "grade_10": {
        "label": "10th Grade",
        "age_range": [15, 16],
        "lix_range": [46, 54],
        "lix_target": 50,
        "vocabulary_notes": "Sophisticated vocabulary. Varied syntax. Precise and concise expression.",
        "sentence_length_avg": 19,
    },
}

WORD_COUNT_PRESETS = {
    400: {"question_count": 4, "label": "Short passage"},
    800: {"question_count": 10, "label": "Long passage"},
}

TEXT_TYPES = {
    "non_fiction": {
        "label": "Non-fiction",
        "formats": [
            "factual explanation",
            "short article",
            "historical account",
            "science explainer",
            "profile or biography",
        ],
    },
    "fiction": {
        "label": "Fiction",
        "formats": [
            "third-person story",
            "first-person narrative",
            "mystery scene",
            "adventure scene",
            "realistic school story",
        ],
    },
    "mixed": {
        "label": "Mixed factual narrative",
        "formats": [
            "fictional story with embedded facts",
            "personal narrative with factual explanations",
            "dialogue-based explanation",
            "journey through a real-world process",
        ],
    },
}

THEMES = {
    "Nature and Environment": [
        "rainforests and biodiversity",
        "oceans and marine life",
        "climate change and its effects",
        "seasons and weather",
        "endangered animals",
        "recycling and sustainability",
        "volcanoes and earthquakes",
        "the water cycle",
        "insects and their roles",
        "national parks and conservation",
    ],
    "Science and Technology": [
        "how the internet works",
        "space exploration and planets",
        "artificial intelligence in everyday life",
        "the human body and health",
        "electricity and energy",
        "robotics and automation",
        "inventions that changed the world",
        "chemistry in the kitchen",
        "the physics of sports",
    ],
    "History and Society": [
        "the Vikings and their voyages",
        "ancient civilisations",
        "democracy and voting",
        "the industrial revolution",
        "famous historical figures",
        "medieval life and castles",
        "the moon landing",
    ],
    "Personal and Social": [
        "friendship and belonging",
        "resilience after failure",
        "online safety and digital life",
        "kindness and empathy",
        "dreams and ambitions",
    ],
    "Sports and Hobbies": [
        "the Olympic Games",
        "learning a new instrument",
        "teamwork in football",
        "cooking and nutrition",
        "outdoor adventures and hiking",
    ],
    "Culture and Arts": [
        "music from around the world",
        "famous paintings and their stories",
        "folktales and mythology",
        "cinema and filmmaking",
        "fashion and cultural identity",
    ],
    "Animals and Pets": [
        "how dogs became human companions",
        "animal communication",
        "migration of birds",
        "unusual animal adaptations",
        "caring for a pet",
    ],
    "Future and Innovation": [
        "future cities",
        "clean energy inventions",
        "space travel",
        "robots at home",
        "new ways of learning",
    ],
    "Health and Body": [
        "sleep and the brain",
        "exercise and the heart",
        "food and energy",
        "how bones heal",
        "why we need water",
    ],
}

MODEL_PRESETS = {
    "OpenAI / ChatGPT default": {
        "provider": "openai",
        "model": "gpt-4o-mini",
        "needs_key": True,
        "key_env": "OPENAI_API_KEY",
        "help": "Default closed-source API option. Paste a key in the sidebar or set OPENAI_API_KEY.",
    },
    "Anthropic Claude": {
        "provider": "anthropic",
        "model": "claude-3-5-haiku-latest",
        "needs_key": True,
        "key_env": "ANTHROPIC_API_KEY",
        "help": "Requires ANTHROPIC_API_KEY.",
    },
    "Google Gemini": {
        "provider": "google",
        "model": "gemini-2.0-flash",
        "needs_key": True,
        "key_env": "GOOGLE_API_KEY",
        "help": "Requires GOOGLE_API_KEY.",
    },
    "Ollama / Llama 3.2 local": {
        "provider": "ollama",
        "model": "llama3.2",
        "needs_key": False,
        "key_env": "",
        "help": "Requires Ollama running locally: ollama serve; ollama pull llama3.2.",
    },
    "Ollama / Mistral local": {
        "provider": "ollama",
        "model": "mistral",
        "needs_key": False,
        "key_env": "",
        "help": "Requires Ollama running locally: ollama serve; ollama pull mistral.",
    },
    "Demo backend / no API": {
        "provider": "demo",
        "model": "demo",
        "needs_key": False,
        "key_env": "",
        "help": "No model call. Useful only for testing UI plumbing and exports.",
    },
}

TEXT_EVAL_DIMS = [
    "age_appropriateness",
    "topic_relevance",
    "engagement",
    "paragraph_consistency",
    "new_concepts_control",
    "british_english",
    "narrative_arc",
    "vocabulary_fit",
]

QUESTION_EVAL_DIMS = [
    "answerable_from_text",
    "single_correct_answer",
    "not_opinion_based",
    "not_general_knowledge",
    "no_inference_required",
    "distractor_quality",
    "question_clarity",
    "text_order",
    "text_reference_accuracy",
    "british_english",
]

DEFAULT_APPROVAL_THRESHOLD = 4.0
DEFAULT_MIN_DIMENSION_SCORE = 3.0
DEFAULT_WORD_TOLERANCE = 20
DEFAULT_MAX_REVISIONS = 4
