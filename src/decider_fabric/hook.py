"""Put a ModelRouter in a Strands agent's path."""

from __future__ import annotations

from dataclasses import dataclass

from .router import RouteDecision, Tier


@dataclass(frozen=True)
class TurnTrace:
    decision: RouteDecision
    tier: Tier
    input_tokens: int
    output_tokens: int
    usd: float
    llm_ms: float


def render(messages) -> str:
    """Flatten Strands messages to `role: text` lines, keeping only text blocks."""
    lines = []
    for message in messages or []:
        for block in message.get("content", []):
            if isinstance(block, dict) and "text" in block:
                lines.append(f"{message.get('role', 'unknown')}: {block['text']}")
    return "\n".join(lines)
