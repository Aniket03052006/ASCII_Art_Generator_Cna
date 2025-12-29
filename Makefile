# Makefile for common development tasks

.PHONY: install dev test lint format web clean

# Install production dependencies
install:
	pip install -e .

# Install all dependencies including dev
dev:
	pip install -e ".[all]"

# Run tests
test:
	pytest tests/ -v

# Run linting
lint:
	ruff check ascii_gen/
	mypy ascii_gen/

# Format code
format:
	black ascii_gen/ tests/ scripts/ web/

# Launch web interface
web:
	python web/app.py

# Clean build artifacts
clean:
	rm -rf build/ dist/ *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

# Train models
train:
	python -c "from ascii_gen.production_training import ProductionCNNMapper, ProductionRFMapper; \
		cnn = ProductionCNNMapper(); cnn.train(); cnn.save('models/production_cnn.pth'); \
		rf = ProductionRFMapper(); rf.train(); rf.save('models/production_rf.joblib')"

# Generate example outputs
examples:
	python tests/test_prompts.py

help:
	@echo "Available commands:"
	@echo "  make install  - Install production dependencies"
	@echo "  make dev      - Install all dependencies"
	@echo "  make test     - Run tests"
	@echo "  make lint     - Run linting"
	@echo "  make format   - Format code"
	@echo "  make web      - Launch web interface"
	@echo "  make train    - Train models"
	@echo "  make clean    - Clean build artifacts"
