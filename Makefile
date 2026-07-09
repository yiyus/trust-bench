.PHONY: test lint coverage

test:
	pytest

lint:
	ruff check .

coverage:
	coverage run -m pytest
	coverage report
