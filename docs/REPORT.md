# decider-fabric: route each agent turn to the cheapest model that can answer it

## The problem

Most agents send every turn to one model. Pick a frontier model and "thanks!" costs as much per token as a proof. Pick a small model and the proof comes back wrong. The fix is to choose the model per turn, and the hard part is who makes that choice.

There are three common answers, and each one costs something:

- **A fixed rule**, such as keyword matching or prompt length. It is free and fast, and it breaks on the first prompt that doesn't fit the rule. "Write a haiku" and "write a lock-free queue" both start with "write".
- **An LLM as the router.** It understands the prompt, but it adds a full model call to every turn. In this project's Region a Haiku 4.5 round trip took 663 ms and a Nova Micro round trip took 358 ms, before the real answer even starts.
- **A trained classifier.** It is fast, but you need labelled data and a training run, and you retrain when the model lineup changes.

decider-fabric uses a fourth option. [Strands Decider 2B](https://github.com/strands-labs/strands-decider) is a 1.9B-parameter decision model. It reads a state and a typed question and returns probabilities over the options you send in the request. It does not generate text. It runs on the laptop, and the option set lives in the request, so changing the model lineup needs no retraining.

## What it does

`ModelRouter` asks the Decider two questions about each new user message, in one HTTP request:

| Question | Type | Options | Used for |
|---|---|---|---|
| `intent` | `choice` | `chat`, `lookup`, `writing`, `code`, `reasoning` | A minimum tier per kind of request |
| `tier` | `score` | One capability description per tier, cheapest first | Which tier answers the turn |

Plain Python then turns the probabilities into a route, in this order:

1. **Cut points on the expected tier.** The router computes `E = sum(i * p_i)` over the tier probabilities. With three tiers and the default cuts `(0.85, 1.3)`, a score below 0.85 goes to the small tier and a score of 1.3 or more goes to the frontier tier.
2. **Intent floors.** `code` and `reasoning` never go below the middle tier.
3. **Hold-down.** The router moves up a tier at once. It moves down only after two turns in a row ask for a lower tier, so one "thanks!" in a hard conversation doesn't drop it to the small model.
4. **Fail upward.** If the Decider server doesn't answer, the turn goes to the top tier. The agent gets slower and costs more, but its answers stay right.

`RoutingHook` puts the router in a Strands agent's path. It listens for `BeforeInvocationEvent`, asks the router for a tier, and sets `agent.model` to that tier's `BedrockModel` before any inference runs. On `AfterInvocationEvent` it records tokens, cost, and latency for the turn. Adding routing to an existing agent takes one argument:

```python
router = ModelRouter(Decider(), LADDERS["anthropic"])
agent = Agent(hooks=[RoutingHook(router, region_name="ap-southeast-2")])
```

## How the pieces fit

```
user message
   │
   ▼
RoutingHook._before ──HTTP POST /v1/systemone──► Decider 2B (local, 127.0.0.1:8099)
   │                  ◄── intent + tier probabilities ──┘
   │ ModelRouter policy: cuts → floors → hold-down
   ▼
agent.model = BedrockModel(tier.model_id)
   │
   ▼
Strands Agent ──HTTPS + SigV4 /model/{id}/converse-stream──► Amazon Bedrock (ap-southeast-2)
   │
   ▼
RoutingHook._after: token delta, cost, latency → TurnTrace
```

| File | Lines | Job |
|---|---|---|
| `src/decider_fabric/decider.py` | 88 | HTTP client for the Decider server. Measures wall-clock latency. |
| `src/decider_fabric/router.py` | 154 | `Tier`, `LADDERS`, the two questions, and the routing policy. |
| `src/decider_fabric/hook.py` | 98 | `RoutingHook`, the Strands wiring, and per-turn cost tracking. |
| `src/decider_fabric/cli.py` | 112 | `dfab route`, `dfab ask`, and `dfab chat`. |
| `tests/test_router.py` | 67 | Six policy tests against a fake Decider. They need no server and no AWS access. |
| `spike/watch_calls.py` | 83 | Prints both network calls of one turn, with secrets redacted. |

Two model ladders ship in `LADDERS`. Every request goes to the ap-southeast-2 endpoint. `au.` inference profiles keep the work in Australia. `apac.` profiles can serve a request from another Asia Pacific Region.

| Ladder | Small | Medium | Frontier |
|---|---|---|---|
| `anthropic` | `apac.amazon.nova-micro-v1:0` | `au.anthropic.claude-haiku-4-5-20251001-v1:0` | `au.anthropic.claude-sonnet-4-6` |
| `nova` | `apac.amazon.nova-micro-v1:0` | `apac.amazon.nova-lite-v1:0` | `apac.amazon.nova-pro-v1:0` |

## Why it looks like a network router

The design copies decisions that IP routers made decades ago.

A router splits its work into a control plane and a data plane. The control plane is slow and general, and it computes routes. The data plane is fast and simple, and it forwards each packet with a table lookup. Here the frontier LLM plays the control plane and Decider 2B plays the data plane, one small decision per turn.

The policy steps have network equivalents too:

- **Intent floors work like QoS marking.** An edge router marks a packet once, for example as voice traffic, and every later hop gives it at least a minimum service level.
- **Hold-down works like BGP route flap damping.** BGP suppresses a route that keeps changing, because each change costs the whole network work. The router here moves up at once and moves down slowly for the same reason.
- **Fail upward works like a default route.** A router with no specific route still forwards traffic upstream to something that can handle it. It never drops the packet.

## What we measured

All numbers come from one MacBook (Apple M5, 16 GB, MLX backend) on 2026-10-20, with model `strands-decider-2B-qwen3.5-v1-2610`.

**Decider latency.** A warm request with two questions and a short state took 155 to 176 ms. The first request at a new input length took 5.7 s, which is a one-off MLX compile. With about 1,500 characters of conversation history in the state, requests took 400 to 530 ms.

**Intent.** On six test prompts, the `intent` confidence was 0.92 to 0.98 for four of them. The other two, a Hamlet summary and a Rust queue design, scored 0.68.

**Tier routing.** With the final rubric, the five test prompts route the same way on both ladders:

| Prompt | E | Tier |
|---|---|---|
| hey, how's it going? | 0.48 | small |
| what's the capital of Peru? | 0.84 | small |
| summarise the plot of Hamlet in three sentences | 0.95 | medium |
| write a python function that reverses a linked list | 1.14 | medium |
| prove there are infinitely many primes of the form 4k+3 | 1.46 | frontier |

**A routed conversation on the `nova` ladder.** This run used the rubric before the third finding below was fixed. It cost $0.00055 over four turns:

| Turn | Decider wanted | Routed to | Cost |
|---|---|---|---|
| prove there are infinitely many primes of the form 4k+3 | nova-pro | nova-pro | $0.00035 |
| thanks! | nova-lite | nova-pro, held 1 of 2 | $0.00018 |
| cool | nova-micro | nova-micro | $0.00001 |
| what's the capital of Peru? | nova-micro | nova-micro | $0.00001 |

Costs use US on-demand list prices. Prices in ap-southeast-2 can differ.

## What we learned

**A cumulative-probability rule never picked the small tier.** The first policy picked the smallest tier whose cumulative probability reached 0.7. The Decider spreads probability across rubric levels even for small talk: "hey, how's it going?" got only 0.68 on the small tier. Every prompt went to the middle tier or higher. The expected score kept the useful property, because probability on higher tiers still pulls the route up, and it fixed the routing.

**Model names in the rubric changed the route.** The first rubric labelled each level with its model name, such as `nova-pro: hard reasoning...`. "What's the capital of Peru?" scored 0.79 on the `anthropic` ladder and 0.87 on the `nova` ladder. The Decider was reading the brand names. The rubric now describes capability only, and both ladders give identical scores.

**A single prefix moved the score.** The hook sent the new message as `user: what's the capital of Peru?`, and `dfab route` sent it without the prefix. Both now send the same text. The Decider responds to every token in its state, so the state format needs the same care as a wire protocol. Fix it once and make every caller send the same bytes.

**Earlier turns raise the score of later ones.** "What's the capital of Peru?" scored lower on its own than after a proof, close to the 0.85 cut. Hold-down hides part of this, but it also cost money. The "thanks!" turn ran on nova-pro for 18 times the nova-micro price.

## What is not done yet

- **No quality measurement.** The tier choices above are checked by eye against six prompts. A labelled prompt set and an LLM judge come next, in `dfab bench`.
- **No comparison against an LLM router.** The latency claim against Haiku as the router is a round-trip estimate, not a benchmark.
- **The `anthropic` ladder answers no turns yet.** Bedrock rejects Anthropic models for the project's AWS account until the Anthropic use-case form is submitted. Routing decisions work on that ladder. End-to-end turns ran only on the `nova` ladder.
- **The cut points are hand-set.** `(0.85, 1.3)` fits the six prompts above, and "capital of Peru" sits 0.01 under the first cut. The bench will fit the cuts to labelled data.
- **One state per decision.** The hook sends up to 1,500 characters of history. Nobody has tested whether less history routes as well and costs less time.
