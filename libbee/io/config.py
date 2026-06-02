"""Validated configuration & credentials for the libbee pipeline (Pydantic settings).

**Where Pydantic belongs in a Polars toolchain:** at the *metadata / orchestration* level, never row-by-row
(converting frames to dicts to validate would defeat Polars' Arrow speed). This module validates the
environment — credentials, the data directory, cache policy — *before* a build runs. Row data is checked
structurally (schema contracts), not per-row, in :mod:`libbee.io.schemas`.

Usage::

    from libbee.io.config import settings
    key = settings().census_api_key           # a pydantic SecretStr | None
    if key:
        do_census_fetch(key.get_secret_value())
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .paths import DATA

DOTENV = ".env"


def _dotenv_value(key: str, env_file: str | Path = DOTENV) -> str | None:
    """Read an **unprefixed** key from a dotenv file. pydantic-settings only reads ``LIBBEE_``-prefixed
    keys from ``.env``, so a plain ``CENSUS_API_KEY=…`` line is invisible to it — this picks it up (the
    same convention the Census adapter has always used). Returns None if the file or key is absent."""
    path = Path(env_file)
    if not path.exists():
        return None
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if line.startswith("#") or "=" not in line:
            continue
        name, _, val = line.partition("=")
        if name.strip() == key:
            return val.strip().strip('"').strip("'") or None
    return None


class LibbeeSettings(BaseSettings):
    """Pipeline configuration, read from ``LIBBEE_*`` env vars / a local ``.env`` (extras ignored).

    ``census_api_key`` also accepts the conventional **unprefixed** ``CENSUS_API_KEY`` — from the environment
    OR straight from ``.env`` — so existing setups keep working without a rename.
    """

    model_config = SettingsConfigDict(env_prefix="LIBBEE_", env_file=DOTENV, env_file_encoding="utf-8", extra="ignore")

    census_api_key: SecretStr | None = None
    data_dir: Path = DATA  # plain Path (not DirectoryPath): valid before the first build creates it
    cache_ttl_seconds: int = 86_400  # 24h

    @field_validator("census_api_key", mode="before")
    @classmethod
    def _fallback_to_unprefixed(cls, v: object) -> object:
        # LIBBEE_CENSUS_API_KEY → unprefixed CENSUS_API_KEY in the environment → unprefixed key in .env.
        return v or os.environ.get("CENSUS_API_KEY") or _dotenv_value("CENSUS_API_KEY")


@lru_cache(maxsize=1)
def settings() -> LibbeeSettings:
    """The validated settings singleton (cached). In tests, call ``settings.cache_clear()`` after
    monkeypatching the environment."""
    return LibbeeSettings()
