"""Rate limiting that forgets who it limited.

The sender's address is needed for a moment — to tell one noisy installation
apart from four thousand quiet ones — and then it must be gone. An IP address
is personal data in most jurisdictions, and it is the single most obvious way
an "anonymous" dataset stops being anonymous.

So the address is never stored. It is hashed with a salt generated fresh when
the process starts, and only the hash is kept. Nothing persists to disk, and a
restart discards every hash. Even someone with full access to the running
service cannot recover the addresses from the counters.
"""

from __future__ import annotations

import hashlib
import os
import secrets
import threading

#: Requests allowed per window, per sender. The client uploads every six hours,
#: so this is generous by two orders of magnitude — it exists to stop abuse,
#: not to police normal use.
DEFAULT_LIMIT = int(os.environ.get("TELEMETRY_RATE_LIMIT", "60"))

#: Length of the window, in seconds.
DEFAULT_WINDOW = float(os.environ.get("TELEMETRY_RATE_WINDOW", "3600"))


class RateLimiter:
    """A fixed-window limiter keyed by a salted hash of the sender."""

    def __init__(self, limit: int | None = None, window_seconds: float | None = None) -> None:
        self.limit = DEFAULT_LIMIT if limit is None else limit
        self.window = DEFAULT_WINDOW if window_seconds is None else window_seconds
        # Fresh per process and never written down, which makes the stored
        # hashes useless to anyone who obtains them — ourselves included.
        self._salt = secrets.token_bytes(16)
        self._lock = threading.Lock()
        self._counters: dict[str, tuple[float, int]] = {}
        self._last_sweep = 0.0

    def _key(self, address: str) -> str:
        return hashlib.blake2b(
            address.encode("utf-8"), key=self._salt, digest_size=16
        ).hexdigest()

    def allow(self, address: str | None, now: float) -> bool:
        """Whether this sender may proceed. The address is not retained."""
        if not address or self.limit <= 0:
            return True

        key = self._key(address)
        with self._lock:
            # Sweep opportunistically, once per window. sweep() existed from
            # the start but nothing in production ever called it — only a
            # test did — so on a public endpoint the table grew one entry per
            # distinct sender for the life of the process.
            if now - self._last_sweep >= self.window:
                self._last_sweep = now
                stale = [k for k, (start, _) in self._counters.items()
                         if now - start >= self.window]
                for stale_key in stale:
                    del self._counters[stale_key]

            window_start, count = self._counters.get(key, (now, 0))
            if now - window_start >= self.window:
                window_start, count = now, 0
            if count >= self.limit:
                return False
            self._counters[key] = (window_start, count + 1)
            return True

    def sweep(self, now: float) -> int:
        """Drop expired windows so the table cannot grow without bound."""
        with self._lock:
            stale = [k for k, (start, _) in self._counters.items() if now - start >= self.window]
            for key in stale:
                del self._counters[key]
            return len(stale)

    @property
    def tracked(self) -> int:
        with self._lock:
            return len(self._counters)
