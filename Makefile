.PHONY: help install lint format test up down logs

help: ## Show this help message
	@echo Usage: make [target]
	@echo.
	@echo Targets:
	@echo   install   Install production and dev dependencies
	@echo   lint      Run ruff check and mypy type checks
	@echo   format    Format code and auto-fix issues with ruff
	@echo   test      Run pytest with coverage report
	@echo   up        Build and start containers in detached mode
	@echo   down      Stop containers preserving volumes
	@echo   logs      View container logs

install: ## Install production and dev dependencies
	pip install --upgrade pip
	pip install -r requirements.txt
	pip install pytest pytest-asyncio pytest-cov ruff mypy

lint: ## Run ruff and mypy
	ruff check .
	mypy src/ --ignore-missing-imports

format: ## Auto-fix and format code with ruff
	ruff check --fix .
	ruff format .

test: ## Run tests with coverage
	pytest tests/ -v --cov=src --cov-report=term-missing

up: ## Start containers via docker compose
	docker compose up -d --build

down: ## Stop containers preserving volumes
	docker compose down

logs: ## View service logs
	docker compose logs -f
