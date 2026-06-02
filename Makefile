# libbee — install, build, test, and demo the data product.
.PHONY: help install build test verify demo explore clean

help:  ## list targets
	@grep -hE '^[a-z-]+:.*##' $(MAKEFILE_LIST) | sed -E 's/:.*## /\t/'

install:  ## install libbee with all extras (editable)
	uv sync --all-extras

build:  ## build the wheel + sdist and validate the metadata
	rm -rf dist
	uv build
	uvx twine check dist/*

test:  ## run the suite (100% coverage gate)
	uv run pytest -q

verify:  ## prove the built data matches the committed MANIFEST fingerprint
	uv run libbee verify

demo:  ## open the libbee.facts() explorer (the bundled example notebook)
	PYTHONPATH=. uv run marimo edit examples/libbee_notebook.py

explore:  ## open the interactive marimo notebook explorer
	PYTHONPATH=. uv run marimo edit examples/libbee_notebook.py

clean:  ## remove build/test artifacts
	rm -rf dist build *.egg-info .pytest_cache .ruff_cache .coverage
