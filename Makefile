# Run from repo root. On Windows without make, use: python scripts/validate_portfolio_project.py

PYTHON ?= python3

.PHONY: validate
validate:
	$(PYTHON) scripts/validate_portfolio_project.py
