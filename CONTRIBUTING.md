# Contributing to Behavioral Consequence Framework

Thank you for your interest in contributing to BCF!

## Development Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/BCF--Behavioral_Consequence_Framework-.git
cd BCF--Behavioral_Consequence_Framework-
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install in development mode:
```bash
pip install -e ".[dev]"
```

## Running Tests

```bash
pytest tests/
```

With coverage:
```bash
pytest --cov=bcf --cov-report=html tests/
```

## Code Style

We use:
- Black for code formatting
- Ruff for linting
- MyPy for type checking

Run before committing:
```bash
black src/ tests/
ruff check src/ tests/
mypy src/
```

## Pull Request Process

1. Create a new branch for your feature
2. Write tests for new functionality
3. Ensure all tests pass
4. Update documentation as needed
5. Submit a pull request with a clear description

## Research Contributions

If adding new behavioral models or frameworks:
1. Include academic references
2. Provide validation against existing literature
3. Add comprehensive tests
4. Update the README with citations

## Questions?

Open an issue on GitHub for any questions or discussions.
