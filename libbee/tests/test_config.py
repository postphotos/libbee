"""Unit tests for libbee.io.config — validated settings & credential fallback.

Each case runs in an isolated cwd (monkeypatch.chdir) so the project's real .env never leaks in; the
dotenv fallback is exercised against a controlled file.
"""

from __future__ import annotations

from pathlib import Path

from libbee.io import config
from libbee.io.config import _dotenv_value
from libbee.io.paths import DATA


def _fresh(monkeypatch, tmp_path, *, env=None, dotenv=None):
    """settings() reading a controlled environment + optional .env, in an isolated cwd."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("CENSUS_API_KEY", raising=False)
    monkeypatch.delenv("LIBBEE_CENSUS_API_KEY", raising=False)
    for k, v in (env or {}).items():
        monkeypatch.setenv(k, v)
    if dotenv is not None:
        (tmp_path / ".env").write_text(dotenv)
    config.settings.cache_clear()
    return config.settings()


def test_defaults(monkeypatch, tmp_path):
    """No env, no .env → no key; data_dir defaults to paths.DATA; 24h cache ttl."""
    s = _fresh(monkeypatch, tmp_path)
    assert s.census_api_key is None
    assert s.data_dir == DATA and isinstance(s.data_dir, Path)
    assert s.cache_ttl_seconds == 86_400


def test_prefixed_key_wins(monkeypatch, tmp_path):
    """LIBBEE_CENSUS_API_KEY is read as a SecretStr (the truthy validator branch)."""
    s = _fresh(monkeypatch, tmp_path, env={"LIBBEE_CENSUS_API_KEY": "prefixed-123"})
    assert s.census_api_key is not None
    assert s.census_api_key.get_secret_value() == "prefixed-123"


def test_unprefixed_env_fallback(monkeypatch, tmp_path):
    """The conventional CENSUS_API_KEY in the environment is picked up."""
    s = _fresh(monkeypatch, tmp_path, env={"CENSUS_API_KEY": "plain-456"})
    assert s.census_api_key is not None
    assert s.census_api_key.get_secret_value() == "plain-456"


def test_unprefixed_from_dotenv(monkeypatch, tmp_path):
    """The unprefixed CENSUS_API_KEY is read straight from .env (pydantic-settings would skip it)."""
    s = _fresh(monkeypatch, tmp_path, dotenv='# comment\nCENSUS_API_KEY="dotenv-789"\nOTHER=x\n')
    assert s.census_api_key is not None
    assert s.census_api_key.get_secret_value() == "dotenv-789"


def test_env_beats_dotenv(monkeypatch, tmp_path):
    """An environment key takes precedence over the .env value."""
    s = _fresh(monkeypatch, tmp_path, env={"CENSUS_API_KEY": "env-wins"}, dotenv="CENSUS_API_KEY=file-loses\n")
    assert s.census_api_key is not None
    assert s.census_api_key.get_secret_value() == "env-wins"


def test_settings_is_cached(monkeypatch, tmp_path):
    """settings() is an lru_cached singleton until cache_clear()."""
    _fresh(monkeypatch, tmp_path)
    assert config.settings() is config.settings()


# ── _dotenv_value branches ───────────────────────────────────────────────────
def test_dotenv_value_missing_file(tmp_path):
    assert _dotenv_value("CENSUS_API_KEY", tmp_path / "nope.env") is None


def test_dotenv_value_parses_quotes_skips_comments_and_blanks(tmp_path):
    p = tmp_path / ".env"
    p.write_text("# header\n\nNOEQUALS\nCENSUS_API_KEY = 'quoted-key' \nEXTRA=y\n")
    assert _dotenv_value("CENSUS_API_KEY", p) == "quoted-key"


def test_dotenv_value_empty_value_is_none(tmp_path):
    p = tmp_path / ".env"
    p.write_text('CENSUS_API_KEY=""\n')
    assert _dotenv_value("CENSUS_API_KEY", p) is None


def test_dotenv_value_absent_key_is_none(tmp_path):
    p = tmp_path / ".env"
    p.write_text("SOMETHING_ELSE=1\n")
    assert _dotenv_value("CENSUS_API_KEY", p) is None
