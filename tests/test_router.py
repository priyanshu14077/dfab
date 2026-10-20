from decider_fabric.decider import Answers
from decider_fabric.router import LADDERS, ModelRouter


class FakeDecider:
    def __init__(self, *replies):
        self.replies = list(replies)

    def ask(self, state, questions):
        intent, probabilities = self.replies.pop(0)
        return Answers(
            answers={
                "intent": {"choice": intent, "confidence": 0.9},
                "tier": {"probabilities": {str(i): p for i, p in enumerate(probabilities)}},
            },
            wall_ms=1.0, server_ms=1.0, input_tokens=10,
        )


def router(*replies, **kwargs):
    return ModelRouter(FakeDecider(*replies), LADDERS["nova"], **kwargs)


def test_spread_out_small_still_routes_small():
    # The decider never puts most mass on one level; E[tier] = 0.47 is still small.
    decision = router(("chat", (0.68, 0.17, 0.15))).decide("", "hey")
    assert (decision.tier, decision.why) == (0, "score")


def test_mass_on_higher_tiers_escalates():
    # argmax says medium, but E[tier] = 1.45 crosses the frontier cut.
    decision = router(("lookup", (0.16, 0.23, 0.61))).decide("", "q")
    assert decision.tier == 2


def test_intent_floor_lifts_code_off_the_smallest_tier():
    decision = router(("code", (0.9, 0.08, 0.02))).decide("", "fix my regex")
    assert (decision.tier, decision.why) == (1, "floor:code")


def test_moves_up_at_once_but_down_only_after_hold_down():
    r = router(
        ("reasoning", (0.0, 0.1, 0.9)),
        ("chat", (0.95, 0.04, 0.01)),
        ("chat", (0.95, 0.04, 0.01)),
        ("reasoning", (0.0, 0.1, 0.9)),
        hold_down=2,
    )
    first, held, released, back_up = (r.decide("", "q") for _ in range(4))
    assert first.tier == 2
    assert (held.tier, held.wanted, held.why) == (2, 0, "hold-down 1/2")
    assert released.tier == 0
    assert back_up.tier == 2
