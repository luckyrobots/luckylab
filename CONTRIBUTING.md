# Contributing to LuckyLab

Thank you for your interest in contributing to LuckyLab! This document provides guidelines and instructions for contributing.

## Development Setup

### Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) package manager
- Git

### Getting Started

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/luckylab.git
   cd luckylab
   ```

2. **Install dependencies with uv**
   ```bash
   uv sync --all-groups
   ```

3. **Install pre-commit hooks**
   ```bash
   uv run pre-commit install
   ```

4. **Verify setup by running tests**
   ```bash
   uv run pytest tests -v
   ```

## Development Workflow

### Code Style

We use [Ruff](https://docs.astral.sh/ruff/) for linting and formatting. The pre-commit hooks will automatically check and fix issues, but you can also run them manually:

```bash
# Check for issues
uv run ruff check src tests

# Fix auto-fixable issues
uv run ruff check --fix src tests

# Format code
uv run ruff format src tests
```

### Running Tests

```bash
# Run all tests
uv run pytest tests -v

# Run specific test file
uv run pytest tests/test_env.py -v

# Run with coverage
uv run pytest tests -v --cov=src/luckylab
```

### Pre-commit Hooks

Pre-commit hooks run automatically on `git commit`. To run them manually:

```bash
# Run on all files
uv run pre-commit run --all-files

# Run on staged files only
uv run pre-commit run
```

## Project Structure

```
luckylab/
├── src/luckylab/
│   ├── configs/          # Configuration dataclasses
│   ├── envs/             # Environment implementations
│   │   └── mdp/          # MDP functions (rewards, terminations, etc.)
│   ├── managers/         # Manager classes (reward, termination, etc.)
│   ├── rl/               # RL training utilities (skrl integration)
│   ├── scripts/          # CLI scripts (train, play, etc.)
│   ├── tasks/            # Task definitions
│   │   └── velocity/     # Velocity tracking task
│   └── utils/            # Utility functions and helpers
├── tests/                # Test suite
└── examples/             # Example scripts
```

## Adding New Features

### Adding a New Task

1. Create a new directory under `src/luckylab/tasks/`:
   ```
   src/luckylab/tasks/my_task/
   ├── __init__.py
   ├── my_task_env_cfg.py
   └── mdp/
       ├── __init__.py
       ├── rewards.py
       ├── terminations.py
       └── observations.py
   ```

2. Define your environment config following the pattern in `velocity_env_cfg.py`

3. Register the task in `src/luckylab/tasks/__init__.py`

4. Add tests in `tests/`

### Adding New Reward/Termination Functions

1. Add your function to the appropriate `mdp/` module
2. The function signature should match existing patterns:
   ```python
   def my_reward(env, **params) -> float:
       """Compute reward based on some criteria."""
       ...
   ```
3. Reference it in your task config using `RewardTermCfg` or `TerminationTermCfg`

### Adding New Algorithms

The RL module uses a builder pattern. To add a new algorithm:

1. Add algorithm-specific config in `src/luckylab/rl/config.py`
2. Add builder function in `src/luckylab/rl/trainer.py`
3. Register in the `_BUILDERS` dict

## Pull Request Process

1. **Create a feature branch**
   ```bash
   git checkout -b feature/my-feature
   ```

2. **Make your changes** and ensure:
   - All tests pass: `uv run pytest tests -v`
   - Code is formatted: `uv run ruff format src tests`
   - No lint errors: `uv run ruff check src tests`
   - Pre-commit passes: `uv run pre-commit run --all-files`

3. **Write meaningful commit messages**
   - Use present tense ("Add feature" not "Added feature")
   - Keep the first line under 72 characters
   - Reference issues when applicable

4. **Push and create a PR**
   ```bash
   git push origin feature/my-feature
   ```

5. **PR Requirements**:
   - Clear description of changes
   - Tests for new functionality
   - Documentation updates if needed
   - All CI checks passing

## Reporting Issues

When reporting issues, please include:

- Python version (`python --version`)
- LuckyLab version (`uv pip show luckylab`)
- Operating system
- Minimal reproducible example
- Full error traceback

## Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help others learn and grow

## Questions?

Feel free to open an issue for questions or join our community discussions.

Thank you for contributing!
