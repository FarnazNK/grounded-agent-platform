.PHONY: install dev lint typecheck test test-unit test-integration eval eval-adversarial eval-ci security docker-build docker-up docker-down migrate benchmark smoke clean

install:
	pip install -e .

dev:
	pip install -e '.[dev,tracing]'

lint:
	ruff check src tests scripts

typecheck:
	mypy src

test: test-unit

test-unit:
	pytest -q -m 'not integration'

test-integration:
	pytest -q -m integration

eval:
	python scripts/run_evals.py --dataset data/eval_datasets/quality.yaml --out benchmarks/results/eval_report.json

eval-adversarial:
	python scripts/run_evals.py --dataset data/eval_datasets/adversarial.yaml --out benchmarks/results/adversarial_eval_report.json

eval-ci: eval eval-adversarial

security:
	pip-audit

migrate:
	alembic upgrade head

benchmark:
	python benchmarks/load/run_benchmark.py --out benchmarks/load/results/local-benchmark.json --markdown-out benchmarks/load/results/local-benchmark.md

smoke:
	python scripts/post_deploy_smoke.py

docker-build:
	docker build -t rag-agent:local .

docker-up:
	docker compose up -d --build

docker-down:
	docker compose down -v

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache build dist *.egg-info coverage.xml .coverage htmlcov
