"""OpenRouter API configuration.

Reads the OpenRouter API key from the environment, transparently loading the
gitignored `.env` file at the repository root when present.  Standard library
only -- no dotenv dependency.

Usage:
    from svtrv2.openrouter import get_openrouter_api_key, openrouter_headers

    headers = openrouter_headers()
    r = requests.get("https://openrouter.ai/api/v1/models", headers=headers)
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
_ENV_FILENAME = ".env"

_loaded = False


def _load_dotenv(path: Path) -> None:
    """Populate missing environment variables from a .env file (no overrides)."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


def _ensure_dotenv_loaded() -> None:
    global _loaded
    if _loaded:
        return
    # Search from the current working directory upward, then fall back to the
    # repository root inferred from this file's location.
    for start in (Path.cwd(), Path(__file__).resolve().parent.parent):
        for candidate in [start, *start.parents]:
            if (candidate / _ENV_FILENAME).is_file():
                _load_dotenv(candidate / _ENV_FILENAME)
                break
        else:
            continue
        break
    _loaded = True


def get_openrouter_api_key() -> str:
    """Return the OpenRouter API key from the environment or `.env` file.

    Raises:
        RuntimeError: if no key is configured.
    """
    _ensure_dotenv_loaded()
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Add it to a .env file at the "
            "repository root (see .env.example) or export it in your shell."
        )
    return key


def openrouter_headers(api_key: str = "") -> Dict[str, str]:
    """Build the standard headers for OpenRouter API requests."""
    key = api_key.strip() or get_openrouter_api_key()
    return {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        # Optional attribution headers recommended by OpenRouter.
        "HTTP-Referer": "https://github.com/Lourdhu02/svtrv2",
        "X-Title": "svtrv2",
    }
