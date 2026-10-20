"""Model tiers and the routing decision."""

from __future__ import annotations

from dataclasses import dataclass


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
