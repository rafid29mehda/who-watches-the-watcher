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

.PHONY: doctor budget pilot agents replay monitors validate-online analyze reproduce-from-cache reproduce-all stop
NICE := nice -n 10

doctor:
	$(PY) wwtw doctor

budget:
	$(PY) wwtw budget

pilot:
	$(NICE) $(PY) wwtw pilot

agents:
	$(NICE) $(PY) wwtw agents

replay:
	$(PY) wwtw replay

monitors:
	$(NICE) $(PY) wwtw monitors

validate-online:
	$(NICE) $(PY) wwtw validate-online

analyze:
	$(PY) wwtw analyze

reproduce-from-cache: analyze

reproduce-all: agents replay monitors validate-online analyze

stop:
	-pkill -INT -f "[w]wtw " ; curl -s -X POST localhost:11434/api/generate -d '{"model":"llama3.2:3b","keep_alive":0}' >/dev/null || true
