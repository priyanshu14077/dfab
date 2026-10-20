# dfab: decider-fabric

Route each turn of a [Strands](https://strandsagents.com) agent to the cheapest Amazon Bedrock model that can answer it. A local decision model, [Strands Decider 2B](https://github.com/strands-labs/strands-decider), makes the choice in about 160 ms with no token cost.

```python
router = ModelRouter(Decider(), LADDERS["anthropic"])
agent = Agent(hooks=[RoutingHook(router, region_name="ap-southeast-2")])
```

Built for the AWS Builder Center Strands Decider 2B hackathon. Decision pattern: `#model-routing`.

## Why route per turn

An agent that sends every turn to one model pays frontier prices for "thanks!" or gets weak answers to proofs. Asking an LLM to pick the model adds a full LLM call to every turn. Decider 2B is a 1.9B-parameter model that returns probabilities over options you send in the request and never generates text, so a routing decision costs one local forward pass.

## How it fits together

```mermaid
flowchart LR
    U([user message]) --> H[RoutingHook<br/>BeforeInvocationEvent]
    H -- "POST /v1/systemone<br/>plain HTTP, localhost" --> D[(Decider 2B<br/>127.0.0.1:8099)]
    D -- "intent + tier probabilities" --> H
    H --> R{ModelRouter<br/>policy}
    R -- small --> M1[Nova Micro]
    R -- medium --> M2[Haiku 4.5 or Nova Lite]
    R -- frontier --> M3[Sonnet 4.6 or Nova Pro]
    M1 & M2 & M3 -- "HTTPS + SigV4<br/>converse-stream" --> B[(Amazon Bedrock<br/>ap-southeast-2)]
    B --> A[RoutingHook<br/>AfterInvocationEvent]
    A --> T[/TurnTrace: tokens, cost, latency/]
```

## What one turn sends over the network

```mermaid
sequenceDiagram
    participant User
    participant Agent as Strands Agent
    participant Hook as RoutingHook
    participant Decider as Decider 2B (local)
    participant Bedrock as Amazon Bedrock

    User->>Agent: "prove there are infinitely many primes"
    Agent->>Hook: BeforeInvocationEvent
    Hook->>Decider: POST /v1/systemone with the state and two questions
    Decider-->>Hook: intent=reasoning 0.92, tier P=[0.16 0.23 0.61]
    Note over Hook: E = 1.46, at or above 1.3, so frontier
    Hook->>Agent: agent.model = BedrockModel(frontier)
    Agent->>Bedrock: POST /model/{id}/converse-stream (SigV4)
    Bedrock-->>Agent: event stream of tokens
    Agent->>Hook: AfterInvocationEvent
    Note over Hook: token delta, cost, latency
    Agent-->>User: answer
```

## How the router picks a tier

One Decider request asks two questions. `intent` is a `choice` over `chat`, `lookup`, `writing`, `code`, and `reasoning`. `tier` is a `score` over one capability description per tier, cheapest first. Plain Python turns the answers into a route:

```mermaid
flowchart TD
    S[Decider answers] --> Q{Decider reachable?}
    Q -- no --> TOP[top tier<br/>fail upward]
    Q -- yes --> E["E = sum of i × p_i"]
    E --> C{"cut points<br/>(0.85, 1.3)"}
    C -- "E < 0.85" --> T0[small]
    C -- "0.85 ≤ E < 1.3" --> T1[medium]
    C -- "E ≥ 1.3" --> T2[frontier]
    T0 & T1 & T2 --> F{intent floor<br/>code or reasoning?}
    F -- "yes, tier below medium" --> FL[raise to medium]
    F -- no --> HD
    FL --> HD{lower than the<br/>current tier?}
    HD -- "no" --> GO[switch now]
    HD -- "yes, fewer than 2 turns in a row" --> HOLD[keep current tier]
    HD -- "yes, 2 turns in a row" --> GO
```

The policy borrows from IP routing. Intent floors work like QoS marking at a network edge. Hold-down works like BGP route flap damping, so the router moves up at once and moves down slowly. Falling back to the top tier works like a default route.

## Project layout

```mermaid
flowchart LR
    subgraph pkg [src/decider_fabric]
        decider[decider.py<br/>HTTP client]
        router[router.py<br/>Tier, LADDERS, ModelRouter]
        hook[hook.py<br/>RoutingHook, TurnTrace]
        cli[cli.py<br/>dfab route, ask, chat]
    end
    tests[tests/test_router.py<br/>policy tests, fake Decider]
    spike[spike/watch_calls.py<br/>prints both network calls]
    docs[docs/REPORT.md<br/>technical report]
    cli --> hook --> router --> decider
    tests --> router
    spike --> hook
```

## Run it

You need an Apple-silicon Mac (or a CUDA or CPU machine), Python 3.12, [uv](https://docs.astral.sh/uv/), and AWS credentials with Bedrock access in `ap-southeast-2`.

1. Install the Decider server and this package:

	```bash
	git clone --depth 1 https://github.com/strands-labs/strands-decider vendor/strands-decider
	uv venv --python 3.12 .venv && source .venv/bin/activate
	uv pip install -e "vendor/strands-decider[mlx]" -e ".[dev]"
	```

2. Start the Decider in a separate terminal. The first start downloads about 4 GB of weights:

	```bash
	strands-decider serve StrandsAgents/strands-decider-2B-qwen3.5-v1-2610 --device mlx --port 8099
	```

3. Route and answer:

	```bash
	export AWS_REGION=ap-southeast-2
	pytest -q tests                                          # policy only, no server or AWS needed
	dfab route "prove there are infinitely many primes"      # the decision only
	dfab --ladder nova ask "what's the capital of Peru?"     # one routed turn
	dfab --ladder nova chat                                  # a routed conversation
	python spike/watch_calls.py "hey there" --ladder nova    # print every network call
	```

If you sign in with `aws login`, botocore needs its CRT extra. This package depends on `botocore[crt]`, so step 1 installs it.

## Model ladders

| Ladder | Small | Medium | Frontier |
|---|---|---|---|
| `anthropic` | `apac.amazon.nova-micro-v1:0` | `au.anthropic.claude-haiku-4-5-20251001-v1:0` | `au.anthropic.claude-sonnet-4-6` |
| `nova` | `apac.amazon.nova-micro-v1:0` | `apac.amazon.nova-lite-v1:0` | `apac.amazon.nova-pro-v1:0` |

To add a ladder, add a tuple of `Tier` objects to `LADDERS` in `src/decider_fabric/router.py`. Describe what each tier is good for in `good_for`, and keep model names out of it. The Decider reads every token in the rubric, and brand names shift the score.

## Status

The `nova` ladder answers end to end. The `anthropic` ladder routes, and its Bedrock calls wait on account access to Anthropic models. The cut points are hand-set from six prompts. `dfab bench`, which fits them to labelled data and compares cost and quality against an LLM router, comes next. Measurements and findings so far are in [the technical report](docs/REPORT.md).

## License

Apache-2.0.
