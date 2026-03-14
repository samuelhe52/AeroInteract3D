UV ?= uv

.PHONY: help setup sync lock run test lint clean

help:
	@echo "Available targets:"
	@echo "  make setup  - Install/update all dependency groups into .venv"
	@echo "  make sync   - Alias for setup"
	@echo "  make lock   - Regenerate uv.lock from pyproject.toml"
	@echo "  make run -- [args] - Run the app entrypoint and forward CLI args"
	@echo "  make test   - Run test suite"
	@echo "  make lint   - Run Ruff lint checks"
	@echo "  make clean  - Remove local caches"

setup:
	$(UV) sync --all-groups

sync: setup

lock:
	$(UV) lock

run:
	$(UV) run python main.py $(filter-out $@,$(MAKECMDGOALS))

%:
	@:

test:
	$(UV) run pytest

lint:
	$(UV) run ruff check .

clean:
	rm -rf .pytest_cache .ruff_cache __pycache__
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
