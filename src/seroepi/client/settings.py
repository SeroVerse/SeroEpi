"""Settings and secrets management for SeroEpi I/O clients and external services."""

import functools
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _parse_credentials_file(path: Path) -> dict[str, str]:
    """Parses a credentials file formatted as key-value pairs or JSON."""
    if not path.is_file():
        return {}
    try:
        content = path.read_text(encoding="utf-8")
        if content.strip().startswith("{"):
            raw = json.loads(content)
            return {k: str(v) for k, v in raw.items()}
        res = {}
        for line in content.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                k, v = line.split("=", 1)
                res[k.strip()] = v.strip().strip("'\"")
        return res
    except Exception:
        return {}


@dataclass(frozen=True, slots=True)
class IOSettings:
    """Configuration and secrets manager for SeroEpi external services and API clients.

    Attributes:
        pathogenwatch_base_url: Base endpoint URL for Pathogenwatch REST API calls.
        pathogenwatch_api_key: Optional API key for authenticated Pathogenwatch requests.
        default_timeout: Default HTTP request timeout in seconds.
    """

    pathogenwatch_base_url: str = "https://pathogen.watch/api"
    pathogenwatch_api_key: str | None = None
    default_timeout: float = 30.0


@functools.lru_cache(maxsize=1)
def get_io_settings() -> IOSettings:
    """Returns the cached singleton ``IOSettings`` instance resolved from environment and credentials.

    Returns:
        Resolved ``IOSettings`` dataclass instance.
    """
    config_dir = Path(os.environ.get("SEROEPI_CONFIG_DIR", Path.home() / ".seroepi"))
    creds_file = config_dir / "credentials"
    creds = _parse_credentials_file(creds_file)

    # 1. API key resolution
    pw_key = (
        os.environ.get("PATHOGENWATCH_API_KEY")
        or os.environ.get("PW_API_KEY")
        or creds.get("PATHOGENWATCH_API_KEY")
        or creds.get("PW_API_KEY")
        or creds.get("pathogenwatch_api_key")
    )

    # 2. Base URL resolution
    pw_url = (
        os.environ.get("PATHOGENWATCH_BASE_URL")
        or os.environ.get("PW_BASE_URL")
        or creds.get("PATHOGENWATCH_BASE_URL")
        or creds.get("PW_BASE_URL")
        or "https://pathogen.watch/api"
    )

    # 3. Timeout resolution
    raw_timeout = os.environ.get("SEROEPI_TIMEOUT") or os.environ.get("SEROEPI_DEFAULT_TIMEOUT")
    timeout = float(raw_timeout) if raw_timeout else 30.0

    return IOSettings(
        pathogenwatch_base_url=pw_url,
        pathogenwatch_api_key=pw_key,
        default_timeout=timeout,
    )


def save_credentials(
    pathogenwatch_api_key: str | None = None,
    config_dir: Path | None = None,
) -> Path:
    """Saves API credentials securely to ``~/.seroepi/credentials`` with ``0600`` permissions.

    Args:
        pathogenwatch_api_key: Optional Pathogenwatch API key string to persist.
        config_dir: Optional directory override. Defaults to ``$SEROEPI_CONFIG_DIR`` or ``~/.seroepi``.

    Returns:
        The ``Path`` to the written credentials file.
    """
    target_dir = config_dir or Path(os.environ.get("SEROEPI_CONFIG_DIR", Path.home() / ".seroepi"))
    target_dir.mkdir(parents=True, exist_ok=True)
    creds_path = target_dir / "credentials"

    existing: dict[str, Any] = {}
    if creds_path.is_file():
        existing = _parse_credentials_file(creds_path)

    if pathogenwatch_api_key is not None:
        existing["PATHOGENWATCH_API_KEY"] = pathogenwatch_api_key

    lines = [f"{k}={v}" for k, v in sorted(existing.items())]
    creds_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    try:
        creds_path.chmod(0o600)
    except OSError:
        pass
    get_io_settings.cache_clear()
    return creds_path
