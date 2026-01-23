# Video Generation

Python monorepo for video generation projects.

## Prerequisites

- Python 3.13+
- [UV](https://docs.astral.sh/uv/) (recommended) or pip

## Setup

### Using UV (Recommended)

```bash
# Install UV if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv sync

# Install pre-commit hooks
uv run pre-commit install
```

### Using pip

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in development mode
pip install -e ".[dev,notebooks]"

# Install pre-commit hooks
pre-commit install
```

## Project Structure

```
video-generation/
├── src/
│   └── video_generation/     # Main Python package
│       └── __init__.py
├── notebooks/                 # Jupyter notebooks
├── scripts/                   # Utility scripts
├── tests/                     # Test files
├── pyproject.toml            # Project configuration
├── .pre-commit-config.yaml   # Pre-commit hooks
└── README.md
```

## Development

### Running Tests

```bash
# With UV
uv run pytest

# With coverage
uv run pytest --cov=src --cov-report=html
```

### Linting and Formatting

```bash
# Run ruff linter
uv run ruff check .

# Run ruff formatter
uv run ruff format .

# Run type checker
uv run mypy src/
```

### Jupyter Notebooks

```bash
# Start Jupyter
uv run jupyter notebook

# Or JupyterLab
uv run jupyter lab
```

## Adding Dependencies

```bash
# Add a runtime dependency
uv add <package>

# Add a dev dependency
uv add --dev <package>
```
