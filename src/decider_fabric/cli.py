"""dfab: watch each turn get routed.

    dfab route "prove there are infinitely many primes"   # the decision only, no Bedrock call
    dfab ask "what's the capital of Peru?"                 # one routed turn
    dfab chat                                              # a routed conversation
"""

from __future__ import annotations

import argparse
import os
import sys

from rich.console import Console

from .decider import Decider, DeciderUnavailable
from .hook import RoutingHook, TurnTrace
from .router import LADDERS, ModelRouter, RouteDecision

console = Console()


def describe(decision: RouteDecision, router: ModelRouter) -> str:
    names = [tier.name for tier in router.ladder]
    if decision.why == "decider-down":
        return f"[red]decider down[/] → [bold]{names[decision.tier]}[/] (fail to top tier)"
    probabilities = " ".join(f"{p:.2f}" for p in decision.probabilities)
    route = f"want={names[decision.wanted]} → [bold]{names[decision.tier]}[/] ({decision.why})"
    return (
        f"[cyan]ingress[/] {decision.intent} {decision.intent_confidence:.2f} │ "
        f"[cyan]route[/] P=[{probabilities}] E={decision.score:.2f} {route} │ "
        f"decider {decision.decider_ms:.0f}ms"
    )


def build_router(args: argparse.Namespace) -> ModelRouter:
    return ModelRouter(
        Decider(args.decider_url), LADDERS[args.ladder],
        cuts=tuple(float(cut) for cut in args.cuts.split(",")), hold_down=args.hold_down,
    )


def run_agent(args: argparse.Namespace, prompts) -> None:
    from strands import Agent

    router = build_router(args)

    def show(trace: TurnTrace) -> None:
        console.print(f"  {describe(trace.decision, router)}", highlight=False)
        console.print(
            f"  [cyan]egress[/] {trace.tier.name} {trace.llm_ms / 1000:.1f}s "
            f"{trace.input_tokens}→{trace.output_tokens} tok ${trace.usd:.5f}",
            highlight=False,
        )

    hook = RoutingHook(router, region_name=args.region, on_turn=show)
    agent = Agent(
        model=hook.model_for(router.ladder[0]),
        hooks=[hook],
        system_prompt="You are a helpful assistant. Answer concisely.",
        callback_handler=None,
    )
    for prompt in prompts:
        result = agent(prompt)
        console.print(f"\n{str(result).strip()}\n", highlight=False)
    if len(hook.turns) > 1:
        total = sum(turn.usd for turn in hook.turns)
        console.print(f"[dim]{len(hook.turns)} turns, ${total:.5f}[/]")


def interactive():
    while True:
        try:
            prompt = console.input("[bold green]you>[/] ").strip()
        except (EOFError, KeyboardInterrupt):
            return
        if prompt in {"exit", "quit"}:
            return
        if prompt:
            yield prompt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dfab", description=__doc__.split("\n")[0])
    parser.add_argument("--ladder", choices=sorted(LADDERS), default="anthropic")
    parser.add_argument("--cuts", default="0.85,1.3", help="expected-tier cut points, one per tier boundary")
    parser.add_argument("--hold-down", type=int, default=2)
    parser.add_argument("--region", default=os.environ.get("AWS_REGION", "ap-southeast-2"))
    parser.add_argument("--decider-url", default=Decider().url)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("route", help="show the routing decision only").add_argument("prompt")
    commands.add_parser("ask", help="answer one routed turn").add_argument("prompt")
    commands.add_parser("chat", help="a routed conversation")
    args = parser.parse_args(argv)

    if args.command == "route":
        router = build_router(args)
        try:
            router.decider.health()
        except DeciderUnavailable as exc:
            console.print(f"[red]{exc}[/]")
            return 1
        console.print(describe(router.decide("", args.prompt), router), highlight=False)
    elif args.command == "ask":
        run_agent(args, [args.prompt])
    else:
        run_agent(args, interactive())
    return 0


if __name__ == "__main__":
    sys.exit(main())
