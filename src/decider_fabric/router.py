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
