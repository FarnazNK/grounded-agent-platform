from __future__ import annotations

import os

import httpx


def main() -> int:
    base_url = os.environ.get("BASE_URL", "http://localhost:8000")
    response = httpx.get(f"{base_url}/health/live", timeout=5)
    response.raise_for_status()
    print(response.json())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
