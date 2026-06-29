.PHONY: setup setup-cpu setup-maniskill test lint fmt gpu-check maniskill-check smoke docs-status clean

setup:
	uv sync --extra cu129 --group dev

setup-cpu:
	uv sync --extra cpu --group dev

setup-maniskill:
	uv sync --extra cu129 --extra maniskill --group dev --group notebooks

test:
	uv run pytest

lint:
	uv run ruff check .

fmt:
	uv run ruff format .
	uv run ruff check . --fix

gpu-check:
	uv run python scripts/check_gpu.py

maniskill-check:
	uv run python scripts/check_maniskill.py

smoke:
	uv run python scripts/smoke_imports.py

docs-status:
	@sed -n '1,200p' docs/status.md

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
