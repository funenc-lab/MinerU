# Repository Guidelines

## Project Structure & Module Organization
`mineru/` contains the Python package. Keep parsing logic in `mineru/backend/` (`pipeline`, `hybrid`, `vlm`), command-line and service entry points in `mineru/cli/`, shared schemas and IO helpers in `mineru/data/`, model wrappers in `mineru/model/`, and cross-cutting helpers in `mineru/utils/`. Put regression tests under `tests/unittest/`; small sample inputs already live in `tests/unittest/pdfs/` and `demo/`. User-facing documentation is built from `docs/en/` and `docs/zh/`, with site configuration in `mkdocs.yml`. Docker deployment files live in `docker/`.

## Build, Test, and Development Commands
Use Python 3.10-3.13. Prefer `uv` for local setup, matching the README and CI.

```bash
uv pip install -e .[test]     # editable install with test deps
pytest tests/unittest/test_e2e.py
coverage run && python tests/get_coverage.py
python -m build --wheel
mkdocs serve
mineru -p demo/pdfs/demo1.pdf -o output/
```

`pytest` runs the end-to-end regression path and writes HTML coverage to `htmlcov/`. `mkdocs serve` requires `pip install -r docs/requirements.txt`.

## Coding Style & Naming Conventions
Follow existing Python style: 4-space indentation, `snake_case` for functions and modules, `PascalCase` for classes, and explicit imports grouped by standard library, third-party, then local modules. Keep new CLI commands under `mineru/cli/` and mirror backend-specific behavior in the matching backend package. Favor small helpers over large mixed-responsibility functions.

## Testing Guidelines
Add or update `pytest` coverage for behavior changes, especially parsing regressions. Extend `tests/unittest/test_e2e.py` when output contracts change, and keep fixture files small and deterministic. Name new tests `test_<feature>.py` or `test_<behavior>()`. Run the target test locally before opening a PR.

## Commit & Pull Request Guidelines
Recent history uses conventional prefixes such as `feat:` and `fix:` with short imperative summaries, for example `feat: improve DOCX package normalization`. Keep commits scoped to one change. PRs should include the motivation, affected input types or backends, local test evidence, and screenshots only when UI or docs rendering changes. Link the related issue when applicable.

## Documentation & Dependency Notes
When changing CLI behavior, APIs, or docs, update both language trees when needed: `docs/en/` and `docs/zh/`. For library- or framework-specific questions during development, fetch current references with `ctx7` before relying on memory.
