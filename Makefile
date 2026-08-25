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

expected:
	$(PY) -c "from ruck.audit import audit; audit('tests/fixtures/leaky_workload', out='demo/expected/audit.md', use_hindsight=False, generated_at='(regenerate with make expected)')"
	$(PY) -m ruck.cli demo --out-dir runs/demo
	cp runs/demo/report.pdf runs/demo/clips.csv demo/expected/
	cp runs/demo/audit.md demo/expected/fixtures_audit.md
