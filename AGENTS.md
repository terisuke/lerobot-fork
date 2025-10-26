# Repository Guidelines

## Project Structure & Module Organization
- `src/lerobot/` holds the Python package, organized by domain (`datasets`, `policies`, `envs`, `robots`, `scripts`). Add new modules beside similar peers and expose them through `src/lerobot/__init__.py`.
- `tests/` mirrors the package layout; reuse existing fixtures in `tests/fixtures/` and keep large assets in `tests/artifacts/`.
- `examples/`, `docs/`, and `benchmarks/` provide runnable notebooks, SSG docs, and performance scripts; update them whenever your change introduces new user-facing behavior.
- Assets and pretrained checkpoints live under `media/` and `outputs/`; avoid committing new large binaries without coordinating on the Hub.

## Build, Test, and Development Commands
- Create a dev environment with `uv sync --extra dev --extra test` (or `poetry sync --extras "dev test"`); both resolve Python 3.10 dependencies declared in `pyproject.toml`.
- Install editable sources via `uv pip install -e .` when iterating on low-level modules without a full sync.
- Run fast validation with `pytest tests/<area>`; use `pytest -m "not slow"` to exclude hardware-heavy suites.
- Execute integration smoke tests with `make test-end-to-end DEVICE=cpu`; switch `DEVICE=gpu` before pushing CUDA-sensitive changes.
- Package-level CLIs such as `lerobot-train` and `lerobot-eval` are exposed after installation—use them to verify pipelines end-to-end.

## Coding Style & Naming Conventions
- Follow Ruff configuration (line length 110, double quotes, space indentation); run `ruff format .` before committing.
- Lint with `ruff check . --fix` and address import ordering enforced by the bundled isort rules.
- Use `snake_case` for modules/functions, `PascalCase` for classes, and `UPPER_CASE` for constants. Prefer Google-style docstrings and type hints on public APIs.

## Testing Guidelines
- Author new tests with `pytest` naming (`test_*.py`, `Test*` classes) adjacent to the feature under `tests/`.
- Gate GPU-, MPS-, or RealSense-dependent scenarios with `pytest.mark.skipif` patterns already present in `tests/processor/*.py`.
- Aim to cover regression paths and update `tests/test_available.py` when registering new policies, datasets, or environments.

## Commit & Pull Request Guidelines
- Keep commits small, present-tense, and imperative (e.g., `Add depth fusion hook`), matching existing history.
- Open PRs from feature branches, link issues, and include a concise summary, test plan, and hardware context (CPU/GPU/OS).
- Attach screenshots or CLI logs when UX, docs, or dataset tooling changes; flag breaking changes in the description and ping reviewers early for robotics-critical updates.
