Python library for parameter and simulation management. Built with hatchling, depends on numpy and sciris.

## Commands
- Test all: `pytest`
- Test single: `pytest tests/test_parameters.py::test_name -x`
- Format: `ruff format src tests`
- Lint: `ruff check src tests`
- Lint fix: `ruff check --fix src tests`

## Testing
- Framework: pytest, tests in `tests/`.
- Four test modules: `test_parameters.py` (Study + Manager), `test_cache.py`, `test_results.py`, `test_cache_integration.py`.
- Shared fixtures in `conftest.py`.

## Code conventions
- Ruff enforces style; config in `.ruff.toml` (line-length 120, preview mode).

## Architecture pointers
- Public API surface: `src/parsimmon/__init__.py`
- Core classes: `Manager` (manager.py), `Study` + `Trial` (study.py), `Results` (results.py)
- Supporting modules: `ranges.py` (range helpers), `_utils.py` (shared internal helpers), `cache.py`
- Full API docs and usage examples: `README.md`
- LLM integration guide for downstream projects: `docs/llm_install.md`

## Git workflow
Conventional commits: `feat:`, `fix:`, `chore:`, breaking changes use `!` suffix (e.g. `rm!:`).

## Boundaries
- **Always**: run `ruff format src tests` and `pytest` before committing.
- **Ask first**: adding new runtime dependencies to `[project.dependencies]`.

## Gotchas
- src layout: package is `src/parsimmon/`, not top-level. Hatchling maps this via `[tool.hatch.build.targets.wheel]`.
- `sciris` is a required dep; `sc.save`/`sc.load` are the default serializers in `cache.py`.
