"""A small synchronous client for a strands-decider server (POST /v1/systemone).

Synchronous on purpose: Strands calls plain-`def` hook callbacks directly, and the server is
best driven one request at a time.
"""

from __future__ import annotations

import os
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
