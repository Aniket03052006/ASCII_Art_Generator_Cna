# Contributing to ASCII Art Generator

Thank you for your interest in contributing! This document provides guidelines and instructions.

## Development Setup

```bash
# Clone the repository
git clone https://github.com/Aniket03052006/ASCII_Art_Generator_Cna.git
cd ASCII_Art_Generator_Cna

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[all]"
```

## Code Style

- **Formatter**: Black (line length 100)
- **Linter**: Ruff
- **Type Checking**: MyPy

Run before committing:
```bash
black ascii_gen/
ruff check ascii_gen/
mypy ascii_gen/
```

## Running Tests

```bash
pytest tests/ -v
```

## Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Make your changes
4. Run tests and linting
5. Commit with clear messages
6. Push and create a Pull Request

## Project Structure

```
ascii_gen/           # Core library code
tests/               # Test suite
scripts/             # CLI tools
web/                 # Web interface
notebooks/           # Jupyter notebooks
models/              # Saved model weights
examples/            # Example files
docs/                # Documentation
```

## Questions?

Open an issue on GitHub!
