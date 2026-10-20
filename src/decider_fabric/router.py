"""The routing decision: which tier of model should answer this turn.

One decider request asks two questions about the turn:

- `intent` (choice): what kind of request it is. Like QoS marking at a network edge, the
  class sets a floor on the tier, so "code" never lands on the smallest model.
- `tier` (score over an ordered rubric): the least capable tier that would answer well.

The policy is plain code over those numbers:

1. Cut points on the expected tier, sum(i * p_i), which the server returns as `score`.
   Probability on higher tiers pulls the score up, so uncertainty pushes the pick upward,
   but no single tier has to win most of the mass: the decider spreads probability over
   rubric levels even for "hey, how's it going?", so a cumulative-probability bar never
   picked the small tier.
2. Intent floor.
3. Flap damping: move up at once, move down only after `hold_down` consecutive turns that
   want a lower tier, so one "thanks!" doesn't drop a hard conversation to the small model.
4. Decider unavailable: route to the top tier, so the agent stays correct, just pricier.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from .decider import Answers, Decider, DeciderUnavailable


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
    hold_down: int = 2
    floors: Mapping[str, int] = field(default_factory=lambda: dict(DEFAULT_FLOORS))
    history_chars: int = 1500
    current: int | None = None
    _lower_streak: int = 0

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

    def __post_init__(self) -> None:
        if len(self.cuts) != len(self.ladder) - 1:
            raise ValueError(f"{len(self.ladder)} tiers need {len(self.ladder) - 1} cut points")

    def decide(self, history: str, prompt: str) -> RouteDecision:
        top = len(self.ladder) - 1
        try:
            answers = self.decider.ask(self.state(history, prompt), self.questions())
        except DeciderUnavailable:
            self.current, self._lower_streak = top, 0
            return RouteDecision(top, top, "decider-down", None, None, (), None, 0.0)

        intent = answers["intent"]
        tier_answer = answers["tier"]
        probabilities = tuple(
            tier_answer["probabilities"][str(i)] for i in range(len(self.ladder))
        )
        score = sum(i * p for i, p in enumerate(probabilities))
        wanted, why = sum(score >= cut for cut in self.cuts), "score"
        floor = min(self.floors.get(intent["choice"], 0), top)
        if floor > wanted:
            wanted, why = floor, f"floor:{intent['choice']}"
        tier, why = self._damp(wanted, why)
        return RouteDecision(
            tier, wanted, why, intent["choice"], intent["confidence"], probabilities, score,
            answers.wall_ms,
        )

    def _damp(self, wanted: int, why: str) -> tuple[int, str]:
        if self.current is None or wanted >= self.current:
            self.current, self._lower_streak = wanted, 0
            return wanted, why
        self._lower_streak += 1
        if self._lower_streak >= self.hold_down:
            self.current, self._lower_streak = wanted, 0
            return wanted, why
        return self.current, f"hold-down {self._lower_streak}/{self.hold_down}"
