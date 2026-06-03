"""`python -m libbee` / `libbee` CLI — build, verify, or list the source adapters."""

from __future__ import annotations

import os
import sys

from . import adapters, build, export_all, export_table, status, verify
from .io.inspect import TableHandle
from .io.paths import ROOT
from .io.store import ExportFormat, tables


def _option(args: list[str], flag: str, default: str | None = None) -> str | None:
    if flag not in args:
        return default
    idx = args.index(flag)
    return args[idx + 1] if idx + 1 < len(args) else default


def _has_census_api_key() -> bool:
    """Check if CENSUS_API_KEY is set in environment or .env file.

    The key is **dev-only**: it's needed solely to (re)build the optional `county_equity`
    table from the Census API. Normal use (loading prebuilt data) never touches it.
    """
    if os.environ.get("CENSUS_API_KEY"):
        return True
    env_path = ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                if key.strip() == "CENSUS_API_KEY" and value.strip():
                    return True
    return False


def _export_targets(args: list[str]) -> list[str]:
    names: list[str] = []
    skip_next = False
    for arg in args[1:]:
        if skip_next:
            skip_next = False
            continue
        if arg in {"--format", "--out"}:
            skip_next = True
            continue
        if arg == "--all":
            continue
        names.append(arg)
    return names


def _export_format(args: list[str]) -> ExportFormat:
    value = (_option(args, "--format", "csv") or "csv").lower()
    if value == "csv":
        return "csv"
    if value == "json":
        return "json"
    raise SystemExit(f"Unsupported export format: {value}")


def main() -> int:
    args = sys.argv[1:]
    if "-h" in args or "--help" in args or "help" in args:
        print(
            "Usage: libbee <command> [options]\n\n"
            "Commands:\n"
            "  build [--force] [--quiet]  Build and conform all data tables (default command)\n"
            "  status             Report whether the local cache is built and ready (exit 0 if so)\n"
            "  list [table]       List published tables, or the columns of one table\n"
            "  verify             Verify that local conformed tables match MANIFEST.json fingerprint\n"
            "  adapters           List all registered data source adapters and their provenance\n"
            "  export [tables...] Export one or more conformed tables to CSV or JSON\n"
            "                     Options:\n"
            "                       --all              Export all conformed tables\n"
            "                       --format csv|json  Specify output format (default: csv)\n"
            "                       --out <path>       Specify output directory or file path\n"
            "  demo [--out <path>] Copy the bundled marimo notebook to the current directory\n"
            "                     (or <path>) and open it with 'marimo edit'.\n\n"
            "Environment Variables:\n"
            "  LIBBEE_DATA        Directory where conformed Parquet tables and DBs are cached\n"
            "  CENSUS_API_KEY     Dev-only. Needed ONLY to (re)build the optional 'county_equity'\n"
            "                     table; not required to load prebuilt data or use the library.\n"
        )
        return 0
    cmd = args[0] if args else "build"
    if cmd == "status":
        s = status()
        print(s.summary())
        for t in s.present:
            print(f"  • {t}")
        if s.missing:
            print(f"  missing: {', '.join(s.missing)}")
        return 0 if s else 1
    if cmd == "list":
        targets = [a for a in args[1:] if not a.startswith("-")]
        if targets:
            handle = TableHandle(targets[0])
            if not handle.exists():
                available = ", ".join(tables()) or "(none — run 'libbee build')"
                print(f"no table '{targets[0]}'. available: {available}")
                return 1
            rows, cols = handle.shape
            print(f"{handle.name} — {rows:,} rows × {cols} cols")
            for name, dtype in handle.schema().items():
                print(f"  {name:<24} {dtype}")
            return 0
        names = tables()
        if not names:
            print("no tables built — run 'libbee build'")
            return 1
        for n in names:
            print(n)
        return 0
    if cmd == "verify":
        return 0 if verify() else 1
    if cmd == "adapters":
        for a in adapters:
            print(f"{a}\n    {a.provenance}")
        return 0
    if cmd == "export":
        file_format = _export_format(args)
        destination = _option(args, "--out")
        if "--all" in args:
            export_all(file_format=file_format, destination=destination)
            return 0
        targets = _export_targets(args)
        if not targets:
            return 1
        export_table(targets[0], file_format=file_format, destination=destination)
        return 0
    if cmd == "demo":
        import importlib.resources
        import importlib.util
        import shutil
        import subprocess
        from pathlib import Path

        if importlib.util.find_spec("marimo") is None:
            print("Error: marimo is not installed. Please install it with: pip install 'libbee[notebook]'")
            return 1

        if not tables():
            print("No conformed data found. Building the cache first (this may take a minute)...")
            build(verbose=True)

        dest = _option(args, "--out")
        notebook_file = Path(dest) if dest else Path("libbee_notebook.py")

        if not notebook_file.exists():
            print(f"Copying bundled example notebook to {notebook_file}...")
            try:
                ref = importlib.resources.files("libbee.examples").joinpath("libbee_notebook.py")
                with ref.open("rb") as src, open(notebook_file, "wb") as dst:
                    shutil.copyfileobj(src, dst)
            except Exception as e:
                print(f"Warning: Could not copy example notebook: {e}")
                print("Proceeding by launching marimo edit with the bundled read-only copy...")
                # notebook_file = Path(importlib.resources.files("libbee.examples").joinpath("libbee_notebook.py"))
                notebook_file = Path(__file__).parent / "examples" / "libbee_notebook.py"

        print(f"Launching marimo edit {notebook_file}...")
        import os

        env = os.environ.copy()
        pkg_parent = str(Path(__file__).parent.parent.resolve())
        if "PYTHONPATH" in env:
            env["PYTHONPATH"] = f"{pkg_parent}{os.pathsep}{env['PYTHONPATH']}"
        else:
            env["PYTHONPATH"] = pkg_parent

        try:
            subprocess.run([sys.executable, "-m", "marimo", "edit", str(notebook_file)], env=env, check=True)
            return 0
        except KeyboardInterrupt:
            return 0
        except Exception as e:
            print(f"Error launching marimo: {e}")
            return 1

    # CENSUS_API_KEY is dev-only — it's needed solely to (re)build the optional
    # 'county_equity' table from the Census API. Every other table (and all
    # prebuilt data) builds without it, so we never block: just say so and skip it.
    if not _has_census_api_key():
        print(
            "ℹ CENSUS_API_KEY not set — that's fine for normal use.\n"
            "  It's only needed in dev to (re)build the optional 'county_equity' table,\n"
            "  which will be skipped. Free key: https://api.census.gov/data/key_signup.html\n"
        )

    quiet = "--quiet" in args
    build(force="--force" in args, verbose=not quiet)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
