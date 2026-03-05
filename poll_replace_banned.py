#!/usr/bin/env python3
"""Poll an order endpoint every 3 seconds until stock is sufficient.

Stops when the response JSON is NOT exactly: {"detail": "Insufficient stock"}

Usage:
  python3 poll_replace_banned.py

Notes:
  - Hardcodes the URL, bearer token, and JSON payload per request.
  - Prints HTTP status + response body each attempt.
"""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request


URL = "https://kiroshop.xyz/shop/api/orders/826/replace-banned"
BEARER_TOKEN = (
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX2lkIjoyMjIsImVtYWlsIjoid2Vpc3Vva2VAZ21haWwuY29tIiwiZXhwIjoxNzcyNjAwNzUzLCJpYXQiOjE3NzIzNDE1NTMsImlzcyI6InBvb2wtc2VydmVyIiwiYXVkIjoicG9vbC1zaG9wIiwianRpIjoiYWVmZmE0MjE4YzFiNDg1Mjk4ZmZjMGMyMzcwNGQyMmYifQ.I6lOckouWKLnpciNd_p3RUvawY5sc7Z_f0rmei17X4k"
)

PAYLOAD = {
    "delivery_id": 1764,
    "account_id": 9864,
}

SLEEP_SECONDS = 2
TIMEOUT_SECONDS = 15


def _post_json(url: str, bearer_token: str, payload: dict) -> tuple[int, str]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={
            "Authorization": f"Bearer {bearer_token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_SECONDS) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            return resp.getcode(), body
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace") if e.fp else str(e)
        return e.code, body


def main() -> int:
    attempt = 0
    target = {"detail": "Insufficient stock"}

    while True:
        attempt += 1
        ts = time.strftime("%Y-%m-%d %H:%M:%S")

        try:
            status, body = _post_json(URL, BEARER_TOKEN, PAYLOAD)
        except urllib.error.URLError as e:
            print(f"[{ts}] attempt={attempt} url_error={e}")
            time.sleep(SLEEP_SECONDS)
            continue
        except Exception as e:  # keep polling on unexpected transient issues
            print(f"[{ts}] attempt={attempt} unexpected_error={type(e).__name__}: {e}")
            time.sleep(SLEEP_SECONDS)
            continue

        print(f"[{ts}] attempt={attempt} status={status} body={body}")

        try:
            parsed = json.loads(body)
        except Exception:
            # Non-JSON response; treat as success condition and stop.
            return 0

        if parsed != target:
            return 0

        time.sleep(SLEEP_SECONDS)


if __name__ == "__main__":
    raise SystemExit(main())
