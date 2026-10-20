"""Put a ModelRouter in a Strands agent's path."""

from __future__ import annotations

from dataclasses import dataclass

from strands.hooks import BeforeInvocationEvent, HookProvider, HookRegistry
from strands.models.bedrock import BedrockModel

from .router import ModelRouter, RouteDecision, Tier


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


class RoutingHook(HookProvider):
    def __init__(self, router: ModelRouter, *, region_name: str | None = None) -> None:
        self.router = router
        self.region_name = region_name
        self._models: dict[str, BedrockModel] = {}

    def register_hooks(self, registry: HookRegistry, **kwargs) -> None:
        registry.add_callback(BeforeInvocationEvent, self._before)

    def model_for(self, tier: Tier) -> BedrockModel:
        if tier.model_id not in self._models:
            self._models[tier.model_id] = BedrockModel(
                model_id=tier.model_id, region_name=self.region_name
            )
        return self._models[tier.model_id]

    def _before(self, event: BeforeInvocationEvent) -> None:
        prompt = render(event.messages)
        history = event.agent.messages
        if history and event.messages and history[-len(event.messages):] == event.messages:
            history = history[: -len(event.messages)]
        decision = self.router.decide(render(history), prompt)
        event.agent.model = self.model_for(self.router.ladder[decision.tier])
