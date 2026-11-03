"""A small per-client rate limiter, so a public demo cannot be used to burn AI or speech credits."""
import threading
import time
from collections import defaultdict, deque

from fastapi import HTTPException, Request


class RateLimiter:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._hits: dict = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, key: str, limit: int, window: float = 60.0) -> bool:
        if not self.enabled:
            return True
        now = time.monotonic()
        with self._lock:
            q = self._hits[key]
            while q and now - q[0] > window:
                q.popleft()
            if len(q) >= limit:
                return False
            q.append(now)
            return True


def client_key(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for", "")   # behind Render's proxy the real address is the first entry
    return (fwd.split(",")[0].strip() if fwd else (request.client.host if request.client else "?"))


def limit(bucket: str, per_minute: int):
    """A FastAPI dependency: at most per_minute calls a minute per client for this bucket."""
    def dep(request: Request):
        limiter = getattr(request.app.state, "limiter", None)
        if limiter and not limiter.check(f"{bucket}:{client_key(request)}", per_minute):
            raise HTTPException(429, "Too many requests, please slow down")
    return dep
