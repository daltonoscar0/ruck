"""Polite, cached HTTP.

One request per second per host, exponential backoff on throttling and server
errors, and every response written to an on-disk cache so a re-run costs
nothing. `ruck demo` runs with ``offline=True``, which turns a cache miss into
an error rather than a request.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlencode, urlsplit

import requests

DEFAULT_CACHE_DIR = Path("data/raw")

# ESPN's edge wants a product token *and* a contact comment: a bare "ruck/0.1"
# gets a 403, and so does a browser impersonation string. The form below is
# both what polite-scraping etiquette asks for and what actually gets served.
# Override with RUCK_USER_AGENT.
DEFAULT_USER_AGENT = "ruck/0.1 (+https://github.com/daltonoscar0/ruck)"

MIN_INTERVAL_SECONDS = 1.0
_last_request_at: dict[str, float] = {}


class OfflineCacheMiss(RuntimeError):
    """Raised when an offline fetcher is asked for something it has not cached."""


class FetchError(RuntimeError):
    """A request failed after every retry."""


@dataclass
class Fetcher:
    """A rate-limited, caching HTTP client."""

    cache_dir: Path = DEFAULT_CACHE_DIR
    offline: bool = False
    min_interval: float = MIN_INTERVAL_SECONDS
    timeout: float = 30.0
    max_retries: int = 4
    user_agent: str = field(default_factory=lambda: os.environ.get(
        "RUCK_USER_AGENT", DEFAULT_USER_AGENT
    ))
    session: requests.Session | None = None

    def __post_init__(self) -> None:
        self.cache_dir = Path(self.cache_dir)
        if self.session is None:
            self.session = requests.Session()
        self.session.headers.update({"User-Agent": self.user_agent, "Accept": "application/json"})

    # -- cache -----------------------------------------------------------

    def cache_path(self, url: str, params: dict[str, Any] | None = None) -> Path:
        full = self._full_url(url, params)
        host = urlsplit(full).netloc or "local"
        digest = hashlib.sha256(full.encode("utf-8")).hexdigest()[:24]
        return self.cache_dir / host / f"{digest}.json"

    def cached(self, url: str, params: dict[str, Any] | None = None) -> bool:
        return self.cache_path(url, params).exists()

    @staticmethod
    def _full_url(url: str, params: dict[str, Any] | None) -> str:
        if not params:
            return url
        joiner = "&" if "?" in url else "?"
        return url + joiner + urlencode(sorted(params.items()))

    # -- fetching --------------------------------------------------------

    def get_json(self, url: str, params: dict[str, Any] | None = None) -> Any:
        """Return parsed JSON for ``url``, from cache when possible."""
        path = self.cache_path(url, params)
        if path.exists():
            return json.loads(path.read_text())
        if self.offline:
            raise OfflineCacheMiss(
                f"offline and {self._full_url(url, params)} is not cached at {path}"
            )
        payload = self._request(self._full_url(url, params))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, separators=(",", ":"), sort_keys=True))
        fetched_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        (path.parent / f"{path.stem}.meta.json").write_text(
            json.dumps({"url": self._full_url(url, params), "fetched_at": fetched_at}, indent=1)
        )
        return payload

    def _request(self, full_url: str) -> Any:
        host = urlsplit(full_url).netloc
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            self._wait_for_slot(host)
            assert self.session is not None
            try:
                response = self.session.get(full_url, timeout=self.timeout)
            except requests.RequestException as exc:
                last_error = exc
            else:
                if response.status_code == 200:
                    try:
                        return response.json()
                    except ValueError as exc:
                        last_error = exc
                        break  # a 200 that is not JSON will not become JSON on retry
                elif response.status_code in (429, 500, 502, 503, 504):
                    last_error = FetchError(f"HTTP {response.status_code} from {full_url}")
                else:
                    raise FetchError(f"HTTP {response.status_code} from {full_url}")
            time.sleep(min(2.0**attempt, 30.0))
        raise FetchError(f"giving up on {full_url}: {last_error}")

    def _wait_for_slot(self, host: str) -> None:
        previous = _last_request_at.get(host)
        now = time.monotonic()
        if previous is not None:
            wait = self.min_interval - (now - previous)
            if wait > 0:
                time.sleep(wait)
        _last_request_at[host] = time.monotonic()
