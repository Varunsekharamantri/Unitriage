"""Token and cost tracking for model calls."""
import threading
from dataclasses import dataclass
from typing import Dict, Mapping, Tuple


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0


# model -> (USD per 1M input tokens, USD per 1M output tokens).
# Models not listed are costed at $0, matching the free tiers this project targets.
# Fill this in from the provider's pricing page if you move to a paid tier.
Prices = Mapping[str, Tuple[float, float]]


class UsageTracker:
    def __init__(self, prices: Prices | None = None):
        self._prices = dict(prices or {})
        self._lock = threading.Lock()
        self._totals: Dict[Tuple[str, str], Dict[str, float]] = {}

    def estimate_cost(self, model: str, usage: Usage) -> float:
        in_price, out_price = self._prices.get(model, (0.0, 0.0))
        return (usage.input_tokens * in_price + usage.output_tokens * out_price) / 1_000_000

    def record(self, provider: str, model: str, usage: Usage) -> None:
        cost = self.estimate_cost(model, usage)
        with self._lock:
            row = self._totals.setdefault(
                (provider, model),
                {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0},
            )
            row["calls"] += 1
            row["input_tokens"] += usage.input_tokens
            row["output_tokens"] += usage.output_tokens
            row["cost_usd"] += cost

    def summary(self) -> dict:
        with self._lock:
            by_model = [
                {"provider": p, "model": m, **row} for (p, m), row in sorted(self._totals.items())
            ]
        return {
            "calls": sum(r["calls"] for r in by_model),
            "input_tokens": sum(r["input_tokens"] for r in by_model),
            "output_tokens": sum(r["output_tokens"] for r in by_model),
            "cost_usd": round(sum(r["cost_usd"] for r in by_model), 6),
            "by_model": by_model,
        }
