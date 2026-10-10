.PHONY: install test test-all lint fmt clean
PY := uv run

install:
	uv sync

test:
	$(PY) pytest -m "not slow" -q

test-all:
	$(PY) pytest -q

lint:
	uv run ruff check . && uv run ruff format --check .

fmt:
	uv run ruff format . && uv run ruff check --fix .

clean:
	rm -rf .venv runs .cache .pytest_cache .ruff_cache .hypothesis
	uv cache prune
