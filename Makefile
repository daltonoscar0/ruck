PY := .venv/bin/python
UV := uv

.PHONY: install test lint fmt demo clean

install:
	$(UV) venv --python 3.11
	$(UV) pip install -e ".[dev]"

test:
	$(PY) -m pytest -q

lint:
	$(UV) run ruff check .
	$(UV) run ruff format --check .

fmt:
	$(UV) run ruff check --fix .
	$(UV) run ruff format .

demo:
	$(PY) -m ruck.cli demo --out-dir runs/demo

clean:
	rm -rf .pytest_cache .ruff_cache runs build dist
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
