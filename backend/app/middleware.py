import time
from collections import defaultdict, deque
from typing import Awaitable, Callable

from fastapi import HTTPException, Request, Response


class RequestSizeLimitMiddleware:
    def __init__(self, app, max_bytes: int):
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope, receive, send):
        if scope.get("type") == "http":
            headers = dict(scope.get("headers", []))
            content_length = headers.get(b"content-length")
            if content_length:
                try:
                    if int(content_length.decode("utf-8")) > self.max_bytes:
                        response = Response(
                            content='{"detail":"Payload too large"}',
                            status_code=413,
                            media_type="application/json",
                        )
                        await response(scope, receive, send)
                        return
                except ValueError:
                    pass
        await self.app(scope, receive, send)


class InMemoryRateLimiter:
    def __init__(self, requests_per_minute: int):
        self.requests_per_minute = requests_per_minute
        self.window_seconds = 60
        self.requests: dict[str, deque[float]] = defaultdict(deque)

    def allow(self, client_id: str) -> bool:
        now = time.time()
        bucket = self.requests[client_id]
        while bucket and now - bucket[0] >= self.window_seconds:
            bucket.popleft()
        if len(bucket) >= self.requests_per_minute:
            return False
        bucket.append(now)
        return True


def get_rate_limit_dependency(
    limiter: InMemoryRateLimiter,
) -> Callable[[Request], Awaitable[None]]:
    async def dependency(request: Request) -> None:
        client_host = request.client.host if request.client else "unknown"
        if not limiter.allow(client_host):
            raise HTTPException(status_code=429, detail="Rate limit exceeded")

    return dependency
