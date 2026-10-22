# Contributing to `SeroEpi`

Contributions are welcome, and they are greatly appreciated! Every improvement — from bug reports and documentation refinements to statistical estimators, Altair plotters, and PyShiny dashboard workflows — helps advance open bacterial sero-epidemiology.

## Types of Contributions

### Report Bugs

Report bugs at https://github.com/SeroVerse/SeroEpi/issues.

If you are reporting a bug, please include:

- Your operating system name, Python version (`>=3.13`), and `jax`/`numpyro`/`polars` versions.
- Minimal cohort data or code snippets sufficient to reproduce the issue.
- Expected statistical or visualization behavior vs. actual output.

### Fix Bugs & Implement Features

Look through the GitHub issues for open bugs or enhancements at https://github.com/SeroVerse/SeroEpi/issues. If you are proposing a new feature:

- Explain how it fits into `SeroEpi`'s architecture (`seroepi.dataset`, `seroepi.variables`, `seroepi.aggregate`, `seroepi.estimators`, `seroepi.plotting`, `seroepi.app`) and ADRs (`docs/adr/`).
- Preserve strict `Variable` identity (`ADR 001`), the `Aggregator` sufficiency-reduction contract (`ADR 002`), and the `StratifiedResultMixin` result interface (`ADR 003`).

## Local Development Setup

`SeroEpi` uses [`uv`](https://docs.astral.sh/uv/) for dependency and environment management.

1. Fork and clone the repository:

```bash
git clone git@github.com:YOUR_USERNAME/SeroEpi.git
cd SeroEpi
```

2. Sync all development, test, and documentation dependency groups:

```bash
uv sync --all-groups --all-extras
```

3. Create a feature branch for your changes:

```bash
git checkout -b feat/your-feature-name
```

## Quality Checks & Testing

Before opening a pull request, verify that formatting, linting, static type checking, unit tests, and strict documentation builds all pass:

```bash
# 1. Lint and check formatting with Ruff
uv run ruff check .
uv run ruff format --check .

# 2. Run static type checking with ty
uv run ty check src/

# 3. Run the pytest test suite
uv run pytest

# 4. Build the Zensical documentation site in strict mode
uv run zensical build --clean --strict
```

## Pull Request Guidelines

1. Include unit tests in `tests/` covering any new estimators, plotters, or bug fixes.
2. Document public classes, methods, and dataclasses using Google-style docstrings so `mkdocstrings` renders complete API references.
3. Ensure all CI checks pass cleanly.
