"""Print every network call one routed turn makes, in order.

    python spike/watch_calls.py "prove there are infinitely many primes" --ladder nova

Leg 1 is the decider (plain HTTP on localhost): the JSON request and response.
Leg 2 is Bedrock (HTTPS + SigV4): botocore's own events show the request before it is
encrypted and the response after it is decrypted. Session tokens are redacted.
"""

from __future__ import annotations

import argparse
import json
import os

from strands import Agent

from decider_fabric import LADDERS, Decider, ModelRouter, RoutingHook

SECRET_HEADERS = {"x-amz-security-token", "authorization"}


def show(title: str, body: object) -> None:
    text = body if isinstance(body, str) else json.dumps(body, indent=2)
    print(f"\n━━ {title}\n{text[:1500]}{' …' if len(text) > 1500 else ''}")


class WatchedDecider(Decider):
    def ask(self, state, questions):
        show(f"[1] POST {self.url}/v1/systemone  (plain HTTP)", {"state": state, "questions": questions})
        answers = super().ask(state, questions)
        show(
            f"[1] ← {answers.wall_ms:.0f}ms wall, {answers.server_ms:.0f}ms in model, "
            f"{answers.input_tokens} input tokens",
            answers.answers,
        )
        return answers


def watch_bedrock(client) -> None:
    def before_send(request, **_):
        headers = {
            k: "<redacted>" if k.lower() in SECRET_HEADERS else v.decode() if isinstance(v, bytes) else v
            for k, v in request.headers.items()
        }
        body = request.body.decode() if isinstance(request.body, bytes) else request.body
        show(f"[2] {request.method} {request.url}  (HTTPS, SigV4-signed)", {"headers": headers})
        show("[2] request body", json.loads(body) if body else "")

    def after_call(http_response, parsed, **_):
        meta = parsed.get("ResponseMetadata", {})
        show(
            f"[2] ← HTTP {http_response.status_code}",
            {"request_id": meta.get("RequestId"), "content-type": http_response.headers.get("content-type")},
        )

    client.meta.events.register("before-send.bedrock-runtime", before_send)
    client.meta.events.register("after-call.bedrock-runtime", after_call)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt")
    parser.add_argument("--ladder", choices=sorted(LADDERS), default="nova")
    args = parser.parse_args()

    router = ModelRouter(WatchedDecider(), LADDERS[args.ladder])
    hook = RoutingHook(router, region_name=os.environ.get("AWS_REGION", "ap-southeast-2"))
    for tier in router.ladder:
        watch_bedrock(hook.model_for(tier).client)

    agent = Agent(model=hook.model_for(router.ladder[0]), hooks=[hook], callback_handler=None)
    result = agent(args.prompt)
    turn = hook.turns[-1]
    show(
        f"[3] routed to {turn.tier.name}: {turn.input_tokens}→{turn.output_tokens} tokens, "
        f"${turn.usd:.5f}, {turn.llm_ms / 1000:.1f}s",
        str(result).strip(),
    )


if __name__ == "__main__":
    main()
