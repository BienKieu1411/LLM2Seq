.PHONY: setup dev lint format docker-up docker-down

setup:
	@echo "Creating virtual environment..."
	python3 -m venv .venv
	@echo "Installing pre-commit hooks..."
	.venv/bin/python -m pip install pre-commit ruff
	.venv/bin/pre-commit install
	@echo "Installing frontend dependencies..."
	cd App/frontend && npm install
	@echo "Installing backend dependencies..."
	.venv/bin/python -m pip install -r App/backend/requirements.txt
	@echo "Setup complete! Remember to activate the environment: source .venv/bin/activate"

dev:
	bash run.sh

lint:
	@echo "Linting frontend..."
	cd App/frontend && npm run lint
	@echo "Linting backend and src..."
	@if [ -f .venv/bin/ruff ]; then .venv/bin/ruff check .; else ruff check .; fi

format:
	@echo "Formatting code..."
	@if [ -f .venv/bin/ruff ]; then .venv/bin/ruff check --fix .; else ruff check --fix .; fi
	@if [ -f .venv/bin/ruff ]; then .venv/bin/ruff format .; else ruff format .; fi

docker-up:
	docker compose -f deploy/docker/docker-compose.yml up --build

docker-down:
	docker compose -f deploy/docker/docker-compose.yml down
