"""Model tiers and the routing decision."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from .decider import Answers, Decider


@dataclass(frozen=True)
class Tier:
    name: str
    model_id: str
    usd_per_mtok_in: float
    usd_per_mtok_out: float
    good_for: str

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        return (input_tokens * self.usd_per_mtok_in + output_tokens * self.usd_per_mtok_out) / 1e6


# On-demand US list prices per million tokens; Sydney (ap-southeast-2) prices can differ, so
# check the Bedrock pricing page before quoting a cost figure.
SMALL = "greetings, small talk, one-line facts, simple rewording or formatting"
MEDIUM = "routine multi-step work: explanations, summaries, standard code snippets, short drafts"
FRONTIER = (
    "hard reasoning: proofs, tricky maths, debugging or designing non-trivial code, "
    "analysis that has to juggle many constraints"
)

LADDERS: dict[str, tuple[Tier, ...]] = {
    "anthropic": (
        Tier("nova-micro", "apac.amazon.nova-micro-v1:0", 0.035, 0.14, SMALL),
        Tier("haiku-4.5", "au.anthropic.claude-haiku-4-5-20251001-v1:0", 1.0, 5.0, MEDIUM),
        Tier("sonnet-4.6", "au.anthropic.claude-sonnet-4-6", 3.0, 15.0, FRONTIER),
    ),
    "nova": (
        Tier("nova-micro", "apac.amazon.nova-micro-v1:0", 0.035, 0.14, SMALL),
        Tier("nova-lite", "apac.amazon.nova-lite-v1:0", 0.06, 0.24, MEDIUM),
        Tier("nova-pro", "apac.amazon.nova-pro-v1:0", 0.80, 3.20, FRONTIER),
    ),
}

INTENTS = {
    "chat": "a greeting, small talk, or a reaction to the previous answer",
    "lookup": "asks for a fact, a definition or a quick piece of information",
    "writing": "asks to write, rewrite, summarise or translate text",
    "code": "asks to write, explain, review or debug code",
    "reasoning": "asks to solve, prove, plan, compare or analyse something that needs careful thought",
}

DEFAULT_FLOORS = {"code": 1, "reasoning": 1}


class Asks(Protocol):
    def ask(self, state: str, questions: Mapping[str, dict]) -> Answers: ...


@dataclass(frozen=True)
class RouteDecision:
    tier: int
    wanted: int
    why: str
    intent: str | None
    intent_confidence: float | None
    probabilities: tuple[float, ...]
    score: float | None
    decider_ms: float


@dataclass
class ModelRouter:
    decider: Asks
    ladder: Sequence[Tier]
    cuts: Sequence[float] = (0.85, 1.3)
    history_chars: int = 1500

    def questions(self) -> dict[str, dict]:
        return {
            "intent": Decider.choice("What kind of request is the user's new message?", INTENTS),
            "tier": Decider.score(
                "What is the least capable assistant that would answer the user's new message well?",
                [f"{tier.name}: {tier.good_for}" for tier in self.ladder],
            ),
        }

    def state(self, history: str, prompt: str) -> str:
        recent = history[-self.history_chars :] if history else "(none)"
        return f"--- EARLIER CONVERSATION ---\n{recent}\n\n--- USER'S NEW MESSAGE ---\n{prompt}"

    def decide(self, history: str, prompt: str) -> RouteDecision:
        answers = self.decider.ask(self.state(history, prompt), self.questions())
        intent = answers["intent"]
        tier_answer = answers["tier"]
        probabilities = tuple(
            tier_answer["probabilities"][str(i)] for i in range(len(self.ladder))
        )
        score = sum(i * p for i, p in enumerate(probabilities))
        wanted = sum(score >= cut for cut in self.cuts)
        return RouteDecision(
            wanted, wanted, "score", intent["choice"], intent["confidence"], probabilities, score,
            answers.wall_ms,
        )
