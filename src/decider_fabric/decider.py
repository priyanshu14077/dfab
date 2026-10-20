"""A small synchronous client for a strands-decider server (POST /v1/systemone).

Synchronous on purpose: Strands calls plain-`def` hook callbacks directly, and the server is
best driven one request at a time.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

DEFAULT_URL = os.environ.get("STRANDS_DECIDER_URL", "http://127.0.0.1:8099")


class DeciderUnavailable(RuntimeError):
    """The decider server did not answer."""


@dataclass(frozen=True)
class Answers:
    answers: dict[str, dict[str, Any]]
    wall_ms: float
    server_ms: float
    input_tokens: int

    def __getitem__(self, key: str) -> dict[str, Any]:
        return self.answers[key]


class Decider:
    """One state, many typed questions, one round trip.

    The state is encoded once and shared by every question, so ask everything a decision
    needs in a single call.
    """

    def __init__(self, url: str = DEFAULT_URL, timeout: float = 30.0) -> None:
        self.url = url.rstrip("/")
        self.timeout = timeout

    def ask(self, state: str, questions: Mapping[str, dict[str, Any]]) -> Answers:
        body = json.dumps({"state": state, "questions": dict(questions)}).encode()
        request = urllib.request.Request(
            f"{self.url}/v1/systemone", data=body, headers={"content-type": "application/json"}
        )
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read())
        except (urllib.error.URLError, TimeoutError) as exc:
            raise DeciderUnavailable(f"no decider at {self.url}: {exc}") from exc
        return Answers(
            answers=payload["answers"],
            wall_ms=(time.perf_counter() - started) * 1000,
            server_ms=payload.get("latency_ms", 0.0),
            input_tokens=payload.get("usage", {}).get("input_tokens", 0),
        )

    def health(self) -> dict[str, Any]:
        try:
            with urllib.request.urlopen(f"{self.url}/health", timeout=5) as response:
                return json.loads(response.read())
        except (urllib.error.URLError, TimeoutError) as exc:
            raise DeciderUnavailable(f"no decider at {self.url}: {exc}") from exc
