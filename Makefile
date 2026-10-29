.PHONY: test lint check

test:
	python -m pytest

lint:
	python -m ruff check .

check: lint test

.PHONY: run
run:
	python -m uvicorn src.app.main:app --reload --host 127.0.0.1 --port 8000
