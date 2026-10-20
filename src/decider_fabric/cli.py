"""dfab: watch each turn get routed.

    dfab route "prove there are infinitely many primes"   # the decision only, no Bedrock call
"""

from __future__ import annotations

import argparse
import sys

from rich.console import Console

from .decider import Decider, DeciderUnavailable
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dfab", description=__doc__.split("\n")[0])
    parser.add_argument("--ladder", choices=sorted(LADDERS), default="anthropic")
    parser.add_argument("--cuts", default="0.85,1.3", help="expected-tier cut points, one per tier boundary")
    parser.add_argument("--hold-down", type=int, default=2)
    parser.add_argument("--decider-url", default=Decider().url)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("route", help="show the routing decision only").add_argument("prompt")
    args = parser.parse_args(argv)

    router = build_router(args)
    try:
        router.decider.health()
    except DeciderUnavailable as exc:
        console.print(f"[red]{exc}[/]")
        return 1
    console.print(describe(router.decide("", args.prompt), router), highlight=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
