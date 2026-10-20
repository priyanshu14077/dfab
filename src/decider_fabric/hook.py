"""Put a ModelRouter in a Strands agent's path.

    router = ModelRouter(Decider(), LADDERS["anthropic"])
    agent = Agent(hooks=[RoutingHook(router, region_name="ap-southeast-2")])

Before each invocation the hook asks the router for a tier and swaps `agent.model` to that
tier's BedrockModel; after it, the hook records tokens, cost and latency for the turn.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

from strands.hooks import AfterInvocationEvent, BeforeInvocationEvent, HookProvider, HookRegistry
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
    def __init__(
        self,
        router: ModelRouter,
        *,
        region_name: str | None = None,
        on_turn: Callable[[TurnTrace], None] | None = None,
    ) -> None:
        self.router = router
        self.region_name = region_name
        self.on_turn = on_turn
        self.turns: list[TurnTrace] = []
        self._models: dict[str, BedrockModel] = {}
        self._pending: tuple[RouteDecision, int, int, float] | None = None

    def register_hooks(self, registry: HookRegistry, **kwargs) -> None:
        registry.add_callback(BeforeInvocationEvent, self._before)
        registry.add_callback(AfterInvocationEvent, self._after)

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
        usage = event.agent.event_loop_metrics.accumulated_usage
        self._pending = (decision, usage["inputTokens"], usage["outputTokens"], time.perf_counter())

    def _after(self, event: AfterInvocationEvent) -> None:
        if self._pending is None:
            return
        decision, input_before, output_before, started = self._pending
        self._pending = None
        usage = event.agent.event_loop_metrics.accumulated_usage
        tier = self.router.ladder[decision.tier]
        input_tokens = usage["inputTokens"] - input_before
        output_tokens = usage["outputTokens"] - output_before
        trace = TurnTrace(
            decision, tier, input_tokens, output_tokens,
            tier.cost(input_tokens, output_tokens), (time.perf_counter() - started) * 1000,
        )
        self.turns.append(trace)
        if self.on_turn:
            self.on_turn(trace)
