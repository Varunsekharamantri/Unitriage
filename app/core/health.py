"""Remembers which AI providers have just failed, so work moves to the ones that are working.

A provider that is out of quota (HTTP 402 or 429) is skipped for a few minutes; one that merely
errored (a dropped connection, a 5xx) for a short while. They stay in the list, at the back, so
if every provider is cooling down they are all still tried, soonest-to-recover first.

Each service (photo triage, note screening, summaries) owns one of these, so a provider that is
rate-limited for photos is not hammered by the next ticket either, and tests don't share state.
"""
import threading
import time
from typing import Callable, Dict, List, Sequence, TypeVar

T = TypeVar("T")


def is_quota_error(exc: BaseException) -> bool:
    """Out of credit or rate-limited: 402/429 from httpx, or Google's RESOURCE_EXHAUSTED."""
    response = getattr(exc, "response", None)
    code = getattr(response, "status_code", None) or getattr(exc, "code", None) or getattr(exc, "status_code", None)
    return code in (402, 429) or "RESOURCE_EXHAUSTED" in str(exc)


class ProviderHealth:
    def __init__(
        self,
        clock: Callable[[], float] = time.monotonic,
        quota_cooldown: float = 300.0,
        error_cooldown: float = 30.0,
    ):
        self._clock = clock
        self._quota = quota_cooldown
        self._error = error_cooldown
        self._until: Dict[str, float] = {}
        self._lock = threading.Lock()

    def failed(self, name: str, exc: BaseException) -> None:
        wait = self._quota if is_quota_error(exc) else self._error
        with self._lock:
            self._until[name] = self._clock() + wait

    def recovered(self, name: str) -> None:
        with self._lock:
            self._until.pop(name, None)

    def cooling(self, name: str) -> bool:
        with self._lock:
            return self._until.get(name, 0.0) > self._clock()

    def order(self, items: Sequence[T], name: Callable[[T], str]) -> List[T]:
        """Working providers first, in their configured order; cooling ones after, soonest first."""
        now = self._clock()
        with self._lock:
            until = {name(i): self._until.get(name(i), 0.0) for i in items}
        ready = [i for i in items if until[name(i)] <= now]
        waiting = sorted((i for i in items if until[name(i)] > now), key=lambda i: until[name(i)])
        return ready + waiting
