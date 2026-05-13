# utils/rate_limiter.py
"""
Async Token Bucket Rate Limiter.

Design: Context manager yang bisa digunakan dengan `async with`.
Thread-safe menggunakan asyncio.Lock.

Digunakan oleh LLMEngine untuk menghormati API rate limits Groq (30 RPM).
"""
import asyncio
import time
from collections import deque


class RateLimiter:
    """
    Sliding window rate limiter.
    
    Lebih accurate dari token bucket sederhana karena
    menggunakan actual timestamps bukan counter.
    """

    def __init__(self, max_calls: int, period_seconds: float):
        self.max_calls = max_calls
        self.period = period_seconds
        self._calls: deque[float] = deque()
        self._lock = asyncio.Lock()

    async def __aenter__(self):
        await self.acquire()
        return self

    async def __aexit__(self, *args):
        pass

    async def acquire(self) -> None:
        """Block hingga slot tersedia dalam rate limit window."""
        async with self._lock:
            now = time.monotonic()

            # Hapus calls yang sudah di luar window
            while self._calls and self._calls[0] <= now - self.period:
                self._calls.popleft()

            # Jika masih penuh, tunggu
            if len(self._calls) >= self.max_calls:
                oldest = self._calls[0]
                sleep_time = self.period - (now - oldest) + 0.01
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                # Bersihkan lagi setelah sleep
                now = time.monotonic()
                while self._calls and self._calls[0] <= now - self.period:
                    self._calls.popleft()

            self._calls.append(time.monotonic())

    @property
    def remaining_calls(self) -> int:
        """Sisa calls yang tersedia di window saat ini."""
        now = time.monotonic()
        active = sum(1 for t in self._calls if t > now - self.period)
        return max(0, self.max_calls - active)