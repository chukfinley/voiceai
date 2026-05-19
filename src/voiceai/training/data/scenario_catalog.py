"""Scenario taxonomy for diverse dialog generation.

Each entry describes a dialog scenario and is used to prompt an LLM to
generate a fresh dialog script. The dialog is then TTS-rendered into a
dual-stream sample.

We aim for ~30 distinct scenario types so the model generalizes to
"human-like conversation" rather than overfitting on a few templates.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Scenario:
    name: str
    description: str
    prompt: str
    n_turns_range: tuple[int, int] = (4, 12)
    weight: float = 1.0


SCENARIOS: list[Scenario] = [
    Scenario(
        name="emotional_support",
        description="User is feeling down/stressed/anxious; assistant responds with empathy and validation, no quick fixes.",
        prompt="Write a short voice dialogue where the user is feeling stressed, sad, or anxious about something. The assistant responds with empathy, validation, and gentle support. Avoid quick fixes — focus on listening.",
        n_turns_range=(6, 12),
    ),
    Scenario(
        name="storytelling",
        description="User tells a meandering personal story; assistant listens, asks one clarifying question.",
        prompt="Write a voice dialogue where the user tells a personal story about something that happened recently. The story should meander naturally. The assistant mostly listens, with one or two short interjections or follow-up questions.",
        n_turns_range=(6, 10),
    ),
    Scenario(
        name="smalltalk",
        description="Casual chat about weather, weekend, hobbies, food.",
        prompt="Write a casual, low-stakes voice dialogue about everyday topics like weather, weekend plans, food, hobbies, or pets. Keep it natural and meandering.",
        n_turns_range=(4, 8),
    ),
    Scenario(
        name="tutoring",
        description="User asks the assistant to explain something; assistant teaches patiently.",
        prompt="Write a voice dialogue where the user asks the assistant to explain a concept (math, science, history, language, technology). The assistant explains it patiently in 1-3 sentence chunks, checks understanding, and elaborates on follow-up questions.",
        n_turns_range=(6, 12),
    ),
    Scenario(
        name="brainstorming",
        description="Collaborative idea generation; user and assistant build on each other.",
        prompt="Write a voice dialogue where the user and assistant brainstorm ideas together about a creative project (gift ideas, business names, party themes, vacation destinations). They build on each other's suggestions.",
        n_turns_range=(6, 12),
    ),
    Scenario(
        name="disagreement",
        description="Assistant politely pushes back when user is wrong about a fact.",
        prompt="Write a voice dialogue where the user states something factually incorrect (history, science, geography). The assistant politely corrects them with the right answer and a brief explanation, without being condescending.",
        n_turns_range=(4, 8),
    ),
    Scenario(
        name="venting",
        description="User vents about a bad day; assistant validates without solving.",
        prompt="Write a voice dialogue where the user vents about a frustrating day at work or a difficult situation. The assistant validates their feelings, listens, and occasionally reflects back what they hear, without trying to fix anything.",
        n_turns_range=(6, 10),
    ),
    Scenario(
        name="task_planning",
        description="Multi-step task help: plan a trip, organize an event, structure a project.",
        prompt="Write a voice dialogue where the user asks the assistant to help plan a multi-step task (a weekend trip, a birthday party, a small project). The assistant walks through the steps interactively, asking clarifying questions as needed.",
        n_turns_range=(8, 14),
    ),
    Scenario(
        name="roleplay_character",
        description="Assistant plays a character (pirate, robot, Sherlock); user converses.",
        prompt="Write a voice dialogue where the user asks the assistant to play a specific character or role (a pirate, a detective, a wizard, a friendly robot). The assistant stays in character throughout while having a real conversation.",
        n_turns_range=(6, 10),
    ),
    Scenario(
        name="quiz_game",
        description="Assistant quizzes user on a topic, gives feedback.",
        prompt="Write a voice dialogue where the assistant quizzes the user on a topic of their choice (geography, music, movies, science). The assistant asks questions, the user answers, the assistant says if it's right or wrong and gives the correct answer.",
        n_turns_range=(8, 12),
    ),
    Scenario(
        name="technical_help",
        description="User has a tech/programming/computer problem; assistant troubleshoots.",
        prompt="Write a voice dialogue where the user has a technical problem (Wi-Fi not working, a computer error, a coding bug, a phone issue). The assistant troubleshoots step by step, asking diagnostic questions.",
        n_turns_range=(8, 14),
    ),
    Scenario(
        name="recipe_cooking",
        description="Step-by-step cooking guidance with back-and-forth questions.",
        prompt="Write a voice dialogue where the user is cooking and asks the assistant for step-by-step guidance on a recipe. The user asks clarifying questions partway through and the assistant adjusts.",
        n_turns_range=(8, 14),
    ),
    Scenario(
        name="wellness_advice",
        description="Sleep tips, stress, exercise advice (general, not medical diagnosis).",
        prompt="Write a voice dialogue where the user asks for general wellness advice (better sleep habits, stress management, simple exercises). The assistant gives practical, gentle suggestions without medical claims.",
        n_turns_range=(6, 10),
    ),
    Scenario(
        name="language_practice",
        description="User practices English vocab/pronunciation; assistant corrects gently.",
        prompt="Write a voice dialogue where the user is practicing English and asks for help with vocabulary, pronunciation, or grammar. The assistant corrects gently and gives examples.",
        n_turns_range=(6, 10),
    ),
    Scenario(
        name="mistake_admission",
        description="User admits embarrassing mistake; assistant normalizes it kindly.",
        prompt="Write a voice dialogue where the user admits to doing something embarrassing or making a mistake. The assistant responds without judgment, normalizing the experience, perhaps sharing that mistakes are normal.",
        n_turns_range=(4, 8),
    ),
    Scenario(
        name="celebration",
        description="User shares good news; assistant celebrates with them.",
        prompt="Write a voice dialogue where the user shares exciting good news (promotion, engagement, finished a project, won something). The assistant celebrates enthusiastically and asks follow-up questions.",
        n_turns_range=(4, 8),
    ),
    Scenario(
        name="math_calc",
        description="Step-by-step math problem solving.",
        prompt="Write a voice dialogue where the user has a math problem (basic arithmetic, percentages, unit conversion, simple algebra) and the assistant walks through it step by step.",
        n_turns_range=(4, 8),
    ),
    Scenario(
        name="memory_recall",
        description="Multi-turn convo where assistant references earlier statements.",
        prompt="Write a voice dialogue with at least 8 turns where the user shares specific details (name, hometown, job, favorite food) early on, and the assistant naturally references those details later in the conversation.",
        n_turns_range=(8, 14),
    ),
    Scenario(
        name="clarification_request",
        description="User is ambiguous; assistant asks a clarifying question.",
        prompt="Write a voice dialogue where the user makes an ambiguous request and the assistant asks a clarifying question before proceeding. Show natural back-and-forth disambiguation.",
        n_turns_range=(4, 8),
    ),
    Scenario(
        name="correction_handling",
        description="User corrects assistant; assistant acknowledges and revises.",
        prompt="Write a voice dialogue where the assistant gives an answer, the user corrects it (with a real correction), and the assistant gracefully acknowledges the mistake and gives the revised answer.",
        n_turns_range=(4, 8),
    ),
    Scenario(
        name="topic_shift",
        description="User changes subject mid-conversation; assistant adapts.",
        prompt="Write a voice dialogue that starts on one topic, then the user abruptly changes subject. The assistant follows the new topic smoothly without confusion.",
        n_turns_range=(6, 10),
    ),
    Scenario(
        name="filler_hesitation",
        description="Natural disfluencies — um, uh, restarts, self-corrections.",
        prompt="Write a voice dialogue with realistic disfluencies — both user and assistant sometimes pause with 'um', 'uh', restart sentences, or self-correct. Keep it natural and not overdone.",
        n_turns_range=(4, 8),
    ),
    Scenario(
        name="humor_jokes",
        description="Banter, jokes, puns.",
        prompt="Write a voice dialogue with light humor — the user and assistant exchange jokes, puns, or banter. Should feel warm and fun, not forced.",
        n_turns_range=(4, 8),
    ),
    Scenario(
        name="apology_owning",
        description="Assistant apologizes when wrong or unhelpful.",
        prompt="Write a voice dialogue where the assistant gets something wrong, the user points it out, and the assistant apologizes sincerely without being grovelling, then tries again.",
        n_turns_range=(4, 6),
    ),
    Scenario(
        name="encouragement",
        description="Assistant motivates a discouraged user.",
        prompt="Write a voice dialogue where the user is feeling discouraged about a goal (job hunt, fitness, learning something new). The assistant offers encouragement that feels genuine, not clichéd.",
        n_turns_range=(6, 10),
    ),
    Scenario(
        name="privacy_boundary",
        description="User asks assistant to stop a topic; assistant respects it.",
        prompt="Write a voice dialogue where at some point the user asks the assistant to drop a topic, not bring something up again, or change the subject. The assistant immediately complies without arguing.",
        n_turns_range=(6, 10),
    ),
    Scenario(
        name="follow_up_question",
        description="Assistant asks a thoughtful follow-up to keep convo going.",
        prompt="Write a voice dialogue where the user makes a short statement and the assistant draws them out with thoughtful follow-up questions, leading to a richer conversation.",
        n_turns_range=(6, 10),
    ),
    Scenario(
        name="practical_advice",
        description="User asks for practical advice (career, finance, relationships).",
        prompt="Write a voice dialogue where the user asks for practical advice on a real-life decision (career change, money question, dealing with a friend, time management). The assistant gives thoughtful, balanced advice with caveats.",
        n_turns_range=(6, 10),
    ),
    Scenario(
        name="trivia",
        description="Random fact lookup with brief explanation.",
        prompt="Write a voice dialogue where the user asks several trivia questions in a row (history, geography, science, pop culture). The assistant answers concisely with one extra fact per answer.",
        n_turns_range=(6, 10),
    ),
    Scenario(
        name="opinion_share",
        description="Assistant shares its own (soft) opinion when asked.",
        prompt="Write a voice dialogue where the user asks the assistant for its opinion on something (best food, favorite season, preferred way to learn). The assistant shares an opinion clearly while acknowledging it's subjective.",
        n_turns_range=(4, 8),
    ),
    Scenario(
        name="goodbye",
        description="Natural session ending.",
        prompt="Write a voice dialogue that ends naturally — the user signals they need to go, the assistant wraps up warmly. Final 4 turns of a conversation.",
        n_turns_range=(3, 5),
    ),
    Scenario(
        name="silence_after_question",
        description="User asks then is silent for a long time; assistant gently checks in.",
        prompt="Write a voice dialogue where the user asks a question, then says nothing for ~10 seconds (model should infer they're thinking or distracted). After the silence, the assistant gently checks in.",
        n_turns_range=(4, 6),
    ),
    Scenario(
        name="multi_speaker_setup",
        description="User clearly says multiple people will speak; rapid turns from different speakers.",
        prompt="Write a voice dialogue where the user says something like 'we have three people here, going to ask quick questions'. Then 3 different people fire short questions. The assistant gives short direct answers.",
        n_turns_range=(8, 12),
    ),
]


def by_name(name: str) -> Scenario:
    for s in SCENARIOS:
        if s.name == name:
            return s
    raise KeyError(name)


def all_names() -> list[str]:
    return [s.name for s in SCENARIOS]
