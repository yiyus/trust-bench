.PHONY: test lint coverage

test:
	pytest -m "not slow"

lint:
	ruff check .

coverage:
	coverage run -m pytest
	coverage report
