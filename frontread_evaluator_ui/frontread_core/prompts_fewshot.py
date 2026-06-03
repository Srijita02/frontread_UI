"""
prompts_fewshot.py — Few-shot prompt templates for FrontRead pipeline.

Structure:
- Text generation: shows 3 exemplar TEXTS ONLY (no questions) — model learns
  format, LIX calibration, vocabulary level, paragraph consistency
- Question generation: shows 3 exemplar TEXT + QUESTION pairs — model learns
  literal question style, text reference format, distractor quality

Exemplars are human-authored FrontRead texts at three LIX levels:
  LOW   — Coffee   (Grade 3, LIX 12.8, 214 words)
  MID   — Rain     (Grade 6, LIX 22.4, 392 words)
  HIGH  — Barbie   (Grade 9, LIX 33.5, 396 words)

INSTRUCTIONS FOR FILLING IN EXEMPLARS:
  Search for <<<PASTE_TEXT_HERE>>> and replace with the body text.
  Search for <<<PASTE_QUESTIONS_HERE>>> and replace with formatted questions.
  Keep the surrounding structure intact.
"""

# ══════════════════════════════════════════════════════════════════════════
# EXEMPLAR TEXTS
# Paste the body text (without title line) between the triple quotes.
# Include the [Word count: NNN] tag as the first line.
# ══════════════════════════════════════════════════════════════════════════

EXEMPLAR_TEXT_LOW = """
Title:
Coffee

Body text:
[Word count: 214, LIX: 12.8, Grade: 3]
You probably have a good idea about what coffee is. Maybe you've even had a little taste? If so, it was probably a bad idea. Coffee tastes sour. It is mostly a drink for adults.

Coffee comes from coffee beans. But they are not real beans at all. Instead they come from a fruit. So coffee actually grows on trees.

How do you make coffee? First you roast the beans. You roast them until they turn dark. Then you crush the beans. Now they turn into a fine powder. Add hot water, and you have coffee.

Today coffee is grown in Asia, South America and Africa. Coffee came to Europe in the 18th century. It came from countries like Yemen and Syria. Coffee had been known there for much longer.

You can get all kinds of coffee. You can buy coffee as a powder. You don't have to grind it yourself. Many people drink this kind of coffee. It doesn't take that long to make.

You can also buy rather expensive capsules with coffee in them. You put them in a machine that makes the coffee for you. Nice and easy.

The next time your mother or father drinks coffee, you can ask them if they know where coffee comes from.
""".strip()


EXEMPLAR_TEXT_MID = """
Title:
Rain

Body text:
[Word count: 392, LIX: 22.4, Grade: 6]
Rain does not fall everywhere. There are deserts where it has not rained for hundreds of years. At the poles, where it is cold, rain turns into snow. Around the equator, it can rain for months on end.

Warm air can contain more water than cold air. This is why rain at the equator is heavier. It is also the reason why it rains in the first place. Warm air is lighter than cold air. It rises as it cools. This separates the water from the air and, since a drop of water is heavier than a drop of air, it falls.

When the drop forms, it consists almost entirely of pure water. On its way down, it picks up dust and other particles. In this way, the air is purified.

There are 4 ways in which the air can be cooled to form a droplet. One has already been mentioned. Namely, when the air rises to a place where it is colder.

Another way is that warm air meets a cold surface and condensation forms. This is the same thing that happens when a car windscreen fogs up or when a coffee pot is closed, and droplets form under its lid. Dew forms. The process that happens in the pot is the same as when cold air from the sea blows over land.

A third way in which air can be cooled is by evaporation. Evaporation happens when the sun heats up water. It may sound strange that the temperature drops when the sun is shining. This is because it takes more energy to turn water into vapour than the sun can provide. When getting out of water or a bath, you will notice that the air seems colder. This is due to evaporation.

The fourth way the air is cooled is by radiation. Radiation is one of the main reasons why the temperature of the planet is not lower. If there was no radiation, it would hardly rain anywhere.

Radiation cannot be seen, but it can often be felt. The way in which the air is cooled by radiation is different from the other three ways. You can get an idea of how this works if you light a fire in winter. The fire is warm. The air around the fire is cold. Yet you can still feel the heat.
""".strip()


EXEMPLAR_TEXT_HIGH = """
Title:
Barbie

Body text:
[Word count: 396, LIX: 33.5, Grade: 9]
You probably know the Barbie doll. It has been the favourite toy of many girls for years. The target group is girls between the ages of 3 and 10.

The iconic doll was invented in 1959. The American woman Ruth Handler realised that her daughter loved to dress up paper dolls. At the time, this was primarily done with paper cutouts. Ruth decided to develop a doll. It would make the game more fun if you had a real character in your hand instead of a piece of paper.

The toy manufacturer Mattel agreed to create the doll. It quickly became popular. In fact, the doll was so successful that an average of two Barbie dolls were sold every second around the world. Clothing and accessories were also produced for the doll. These were sold alongside the doll. You could get endless amounts of clothes for the Barbie doll to wear.

Just two years after the invention of Barbie herself, the producer realised that the doll needed a boyfriend. Ken was created. Just like the Barbie doll, Ken was also available in a variety of clothes and accessories.

Since then, more characters have been created – all related to the original Barbie and all with matching clothes and accessories. In fact, since its invention in 1959, over a billion garments have been produced for Barbie and her friends. The clothes have been designed by over 70 different renowned fashion designers worldwide.

In addition to clothes, Mattel also produces toy animals. Over the years, Barbie has had over 50 different pets. This includes dogs, cats, horses, monkeys, pandas, lions, giraffes and zebras.

The doll has been met with criticism from many sides. Some argue that Barbie projects an unrealistic image of the female body. The narrow waist, narrow hips and thin legs would not be able to move in real life, according to studies. The body measurements are far too extreme, and not to be strived for.

In 2016, the manufacturer took the criticism to heart. Today, Barbies are available in a wider range of body types. There is still a focus on showing little girls that they can be anything they want to be. That's why there's a Presidential Barbie, which was launched in 2012. Furthermore, there are not only white, blonde Barbies. There are also Barbies who are, for example, African American and Asian.
""".strip()


# ══════════════════════════════════════════════════════════════════════════
# EXEMPLAR QUESTIONS
# Each block pairs the text with its questions.
# Questions follow the exact output format the model must produce.
# ══════════════════════════════════════════════════════════════════════════

EXEMPLAR_QA_LOW = """
--- EXAMPLE: Grade 3, LIX 12.8, 214 words, topic: Coffee ---

TEXT:
You probably have a good idea about what coffee is. Maybe you've even had a little taste? If so, it was probably a bad idea. Coffee tastes sour. It is mostly a drink for adults.

Coffee comes from coffee beans. But they are not real beans at all. Instead they come from a fruit. So coffee actually grows on trees.

How do you make coffee? First you roast the beans. You roast them until they turn dark. Then you crush the beans. Now they turn into a fine powder. Add hot water, and you have coffee.

Today coffee is grown in Asia, South America and Africa. Coffee came to Europe in the 18th century. It came from countries like Yemen and Syria. Coffee had been known there for much longer.

You can get all kinds of coffee. You can buy coffee as a powder. You don't have to grind it yourself. Many people drink this kind of coffee. It doesn't take that long to make.

You can also buy rather expensive capsules with coffee in them. You put them in a machine that makes the coffee for you. Nice and easy.

The next time your mother or father drinks coffee, you can ask them if they know where coffee comes from.

QUESTIONS:
1. What happens to the coffee bean before water is added to make coffee?
A: It gets warmed in an oven
B: It gets added sugar
C: It gets mixed with cinnamon
D: It gets roasted and crushed
Correct: D
Text reference: First you roast the beans. You roast them until they turn dark. Then you crush the beans.

2. When did coffee come to Europe?
A: 11th century
B: 14th century
C: 16th century
D: 18th century
Correct: D
Text reference: Coffee came to Europe in the 18th century.

3. From where did Europeans first import coffee?
A: Yemen and Syria
B: Sweden and Norway
C: Russia and China
D: Japan and France
Correct: A
Text reference: It came from countries like Yemen and Syria.

4. What kind of coffee is described as quite expensive?
A: Coffee in powder form
B: Filter coffee
C: Coffee in capsules
D: Coffee from a coffee machine
Correct: C
Text reference: You can also buy rather expensive capsules with coffee in them.
--- END EXAMPLE ---
""".strip()


EXEMPLAR_QA_MID = """
--- EXAMPLE: Grade 6, LIX 22.4, 392 words, topic: Rain ---

TEXT:
Rain does not fall everywhere. There are deserts where it has not rained for hundreds of years. At the poles, where it is cold, rain turns into snow. Around the equator, it can rain for months on end.

Warm air can contain more water than cold air. This is why rain at the equator is heavier. It is also the reason why it rains in the first place. Warm air is lighter than cold air. It rises as it cools. This separates the water from the air and, since a drop of water is heavier than a drop of air, it falls.

When the drop forms, it consists almost entirely of pure water. On its way down, it picks up dust and other particles. In this way, the air is purified.

There are 4 ways in which the air can be cooled to form a droplet. One has already been mentioned. Namely, when the air rises to a place where it is colder.

Another way is that warm air meets a cold surface and condensation forms. This is the same thing that happens when a car windscreen fogs up or when a coffee pot is closed, and droplets form under its lid. Dew forms.

A third way in which air can be cooled is by evaporation. Evaporation happens when the sun heats up water. It may sound strange that the temperature drops when the sun is shining. This is because it takes more energy to turn water into vapour than the sun can provide.

The fourth way the air is cooled is by radiation. Radiation is one of the main reasons why the temperature of the planet is not lower. If there was no radiation, it would hardly rain anywhere.

Radiation cannot be seen, but it can often be felt. The fire is warm. The air around the fire is cold. Yet you can still feel the heat.

QUESTIONS:
1. In how many ways can air be cooled to form a droplet?
A: 3
B: 4
C: 5
D: 6
Correct: B
Text reference: There are 4 ways in which the air can be cooled to form a droplet.

2. What happens when warm air meets a cold surface?
A: It starts to snow
B: Dew is forming
C: It starts to thunder
D: It starts to storm
Correct: B
Text reference: Another way is that warm air meets a cold surface and condensation forms. Dew forms.

3. How can the temperature drop when the sun is shining?
A: The sunlight reflects off the water
B: The sun shines on the clouds
C: The water evaporates
D: Shadows move more rapidly
Correct: C
Text reference: This is because it takes more energy to turn water into vapour than the sun can provide.

4. What is one of the main reasons why the temperature is not lower on Earth?
A: High pressure
B: Mountains
C: Radiation
D: Cloud
Correct: C
Text reference: Radiation is one of the main reasons why the temperature of the planet is not lower.
--- END EXAMPLE ---
""".strip()


EXEMPLAR_QA_HIGH = """
--- EXAMPLE: Grade 9, LIX 33.5, 396 words, topic: Barbie ---

TEXT:
You probably know the Barbie doll. It has been the favourite toy of many girls for years. The target group is girls between the ages of 3 and 10.

The iconic doll was invented in 1959. The American woman Ruth Handler realised that her daughter loved to dress up paper dolls. At the time, this was primarily done with paper cutouts. Ruth decided to develop a doll. It would make the game more fun if you had a real character in your hand instead of a piece of paper.

The toy manufacturer Mattel agreed to create the doll. It quickly became popular. In fact, the doll was so successful that an average of two Barbie dolls were sold every second around the world. Clothing and accessories were also produced for the doll. These were sold alongside the doll.

Just two years after the invention of Barbie herself, the producer realised that the doll needed a boyfriend. Ken was created. Just like the Barbie doll, Ken was also available in a variety of clothes and accessories.

Since then, more characters have been created. In fact, since its invention in 1959, over a billion garments have been produced for Barbie and her friends. The clothes have been designed by over 70 different renowned fashion designers worldwide.

In addition to clothes, Mattel also produces toy animals. Over the years, Barbie has had over 50 different pets. This includes dogs, cats, horses, monkeys, pandas, lions, giraffes and zebras.

The doll has been met with criticism from many sides. Some argue that Barbie projects an unrealistic image of the female body. The body measurements are far too extreme, and not to be strived for.

In 2016, the manufacturer took the criticism to heart. Today, Barbies are available in a wider range of body types. That's why there's a Presidential Barbie, which was launched in 2012. Furthermore, there are not only white, blonde Barbies. There are also Barbies who are, for example, African American and Asian.

QUESTIONS:
1. What is Barbie's target audience?
A: 5-12 year old girls
B: 3-10 year old girls
C: 7-15 year old girls
D: 10-17 year old boys
Correct: B
Text reference: The target group is girls between the ages of 3 and 10.

2. When was the Barbie doll invented?
A: 1959
B: 1961
C: 1990
D: 2008
Correct: A
Text reference: The iconic doll was invented in 1959.

3. Which Barbie was made to show girls that they can be anything they want to be?
A: A police officer Barbie
B: A chief executive officer Barbie
C: A Presidential Barbie
D: An engineer Barbie
Correct: C
Text reference: That's why there's a Presidential Barbie, which was launched in 2012.

4. How many different pets has Barbie had?
A: Over 80
B: Over 50
C: Over 30
D: Over 20
Correct: B
Text reference: Over the years, Barbie has had over 50 different pets.
--- END EXAMPLE ---
""".strip()


# ══════════════════════════════════════════════════════════════════════════
# FEW-SHOT TEXT GENERATION PROMPT
# Shows 3 exemplar TEXTS ONLY — no questions shown here.
# Model learns: format, LIX calibration, vocabulary level, paragraph style.
# ══════════════════════════════════════════════════════════════════════════

FEW_SHOT_TEXT_GEN_SYSTEM = """
You are an expert educational writer for FrontRead, a Scandinavian reading-training platform.
Write in British English throughout (colour, organise, centre, practise, maths, etc.).

You write age-appropriate reading texts for school students. Your writing style matches a
children's author: clear, engaging, with a strong narrative arc even in non-fiction.

STRICT RULES:
- British English only. No American spellings.
- Plain prose only inside the body — no bullet points, no bold, no markdown headers.
- Do NOT introduce more than one new concept or term per text. This is a reading exercise,
  not a lesson. Use vocabulary the student already knows at their grade level.
- Every paragraph must maintain a comparable readability level to the whole text.
  Do not have simple paragraphs mixed with very complex ones.
- Write EXACTLY the number of words specified. Count carefully.
- Output ONLY the structured text in the format shown in the examples. Nothing else.
""".strip()


FEW_SHOT_TEXT_GEN_PROMPT = """
Below are three examples of correctly written reading texts at different difficulty levels.
Study them carefully. Notice how:
- Vocabulary and sentence length increase from the low to the high example
- Each paragraph stays at a comparable difficulty level throughout the text
- The [Word count: NNN] tag appears as the first line of the body
- Plain prose only — no bullet points, no bold, no headers inside the text
- British English is used throughout

=== EXAMPLE 1: Low difficulty (Grade 3, LIX ~13, 214 words) ===

{exemplar_low}

=== EXAMPLE 2: Medium difficulty (Grade 6, LIX ~22, 392 words) ===

{exemplar_mid}

=== EXAMPLE 3: High difficulty (Grade 9, LIX ~34, 396 words) ===

{exemplar_high}

===

Now generate a NEW reading text following the same format.
Do NOT copy or closely paraphrase any example. Write entirely original content on the new topic.

PARAMETERS:
- Grade: {grade_label}
- Topic: {topic}
- Text type: {text_type} ({text_format})
- Target word count: EXACTLY {word_count} words (±20 words tolerance)
- Target LIX score: {lix_target} (acceptable range: {lix_min}–{lix_max})
- Avg sentence length to aim for: ~{sentence_length_avg} words per sentence
- Vocabulary: {vocabulary_notes}
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


# ══════════════════════════════════════════════════════════════════════════
# FEW-SHOT REVISION PROMPT
# Same as zero-shot revision — no exemplars needed here since
# the model is revising its own output with computed feedback.
# Import TEXT_REVISION_PROMPT from prompts.py and use directly.
# ══════════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════════
# FEW-SHOT QUESTION GENERATION SYSTEM PROMPT
# ══════════════════════════════════════════════════════════════════════════

FEW_SHOT_QUESTION_GEN_SYSTEM = """
You are an expert question writer for FrontRead, a Scandinavian reading-training platform.
Write in British English throughout.

You write LITERAL COMPREHENSION questions only. Every question must be directly and
explicitly answered somewhere in the text — the student finds the answer by locating
the relevant sentence or phrase and reading it.

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


# ══════════════════════════════════════════════════════════════════════════
# FEW-SHOT QUESTION GENERATION USER PROMPT
# Shows 3 exemplar TEXT + QUESTION pairs — model learns question style.
# ══════════════════════════════════════════════════════════════════════════

FEW_SHOT_QUESTION_GEN_PROMPT = """
Below are three examples of a reading text paired with correctly written literal
comprehension questions. Study them carefully. Notice how:
- Every answer is explicitly stated in the text — no inference needed
- The Text reference field quotes the exact phrase from the text containing the answer
- Questions follow the order of the text from beginning to end
- Wrong options are plausible but clearly wrong once the relevant sentence is found
- No question requires opinion, analysis, or general knowledge
- British English is used throughout

{exemplar_qa_low}

{exemplar_qa_mid}

{exemplar_qa_high}

===

Now generate {question_count} NEW literal comprehension questions for the text below.
Do NOT copy the example questions. Write entirely new questions for the new text.

TEXT DETAILS:
- Grade: {grade_label}
- Topic: {topic}
- Text word count: {word_count}

THE TEXT:
Title: {title}

{body}

REQUIREMENTS:
- Exactly {question_count} questions
- ALL questions are literal — answer must be explicitly stated in the text
- No inference, no opinion, no general knowledge
- Questions follow the order of the text
- One unambiguous correct answer per question
- Three plausible but wrong distractors
- Text reference quotes the exact phrase containing the answer
- British English throughout

OUTPUT FORMAT — follow exactly:

{question_template}
""".strip()


# ══════════════════════════════════════════════════════════════════════════
# FEW-SHOT PROMPT HELPERS
# ══════════════════════════════════════════════════════════════════════════

def build_few_shot_text_prompt(**params: object) -> str:
    """Return a fully formatted few-shot text generation prompt."""
    return FEW_SHOT_TEXT_GEN_PROMPT.format(
        exemplar_low=EXEMPLAR_TEXT_LOW,
        exemplar_mid=EXEMPLAR_TEXT_MID,
        exemplar_high=EXEMPLAR_TEXT_HIGH,
        **params,
    )


def build_question_template(question_count: int) -> str:
    """Return an explicit numbered output scaffold for question generation."""
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


def build_few_shot_question_prompt(**params: object) -> str:
    """Return a fully formatted few-shot question generation prompt."""
    question_count = int(params.get("question_count", 4))
    return FEW_SHOT_QUESTION_GEN_PROMPT.format(
        exemplar_qa_low=EXEMPLAR_QA_LOW,
        exemplar_qa_mid=EXEMPLAR_QA_MID,
        exemplar_qa_high=EXEMPLAR_QA_HIGH,
        question_template=build_question_template(question_count),
        **params,
    )


TEXT_REVISION_PROMPT = """
You previously wrote a reading text for FrontRead. Python metrics show that it does not yet meet the required constraints.

ORIGINAL TITLE:
{title}

ORIGINAL BODY:
{body}

COMPUTED METRICS:
- Actual word count: {actual_words}
- Target word count: {word_count} ±20
- Actual LIX: {lix_actual}
- Required LIX range: {lix_min}–{lix_max}
- Average sentence length: {avg_sent_len}
- Long-word percentage: {long_word_pct}%

PROBLEMS TO FIX:
{problems}

Revise the text so that it better satisfies the target word count and LIX band.
Keep the same topic and grade level.
Keep British English.
Keep paragraph difficulty consistent.
Do not add markdown, bullets, explanations, or analysis.

OUTPUT FORMAT — use these exact headers:

Title:
[title here]

Body text:
[Word count: NNN]
[Full revised text here — plain prose only]
""".strip()
