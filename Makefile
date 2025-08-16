.PHONY: setup lint fmt test train ablate demo clean help test-sst2 debug-sst2

help:  ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-15s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

setup:  ## Create virtual environment and install dependencies
	python3 -m venv venv
	. venv/bin/activate && pip install --upgrade pip
	. venv/bin/activate && pip install -e ".[dev]"

lint:  ## Run linting checks
	. venv/bin/activate && ruff check src/ tests/
	. venv/bin/activate && black --check src/ tests/

fmt:  ## Format code with black and ruff
	. venv/bin/activate && black src/ tests/
	. venv/bin/activate && ruff check --fix src/ tests/

test:  ## Run tests with coverage
	. venv/bin/activate && python -m pytest tests/ -v --cov=src/mocanet --cov-report=term-missing

train:  ## Train MOCA-Net on copy task (fast CPU run)
	. venv/bin/activate && python -m mocanet.train --config-name=copy_task

ablate:  ## Run ablation studies
	. venv/bin/activate && python -m mocanet.ablation

demo:  ## Run text classification demo
	. venv/bin/activate && python -m mocanet.train --config-name=text_cls

clean:  ## Clean up generated files
	rm -rf __pycache__/
	rm -rf src/**/__pycache__/
	rm -rf tests/**/__pycache__/
	rm -rf .pytest_cache/
	rm -rf .coverage/
	rm -rf htmlcov/
	rm -rf runs/
	rm -rf venv/
	find . -name "*.pyc" -delete
	find . -name "*.pyo" -delete
	find . -name "*.pyd" -delete

install-dev:  ## Install development dependencies
	. venv/bin/activate && pip install -e ".[dev]"

run-copy:  ## Quick copy task training run
	. venv/bin/activate && python -m mocanet.train --config-name=copy_task training.max_steps=1000

run-text:  ## Quick text classification run
	. venv/bin/activate && python -m mocanet.train --config-name=text_cls training.max_steps=500

test-sst2:  ## Test SST-2 dataset loading
	. venv/bin/activate && python scripts/test_sst2.py

debug-sst2:  ## Debug SST-2 data and model
	. venv/bin/activate && python scripts/debug_data.py
