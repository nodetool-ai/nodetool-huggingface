# Contributor Guidelines

This repository provides additional nodes for the [nodetool](https://github.com/nodetool-ai/nodetool) project and depends on [nodetool-core](https://github.com/nodetool-ai/nodetool-core).

## Code Style

- Use **Python 3.11+** syntax.
- All nodes live under `src/nodetool/nodes/huggingface` and must inherit from `BaseNode`.
- Node attributes are defined with `pydantic.Field` and async `process` methods should return the appropriate reference type.
- Each node must contain a short docstring describing the model and several example use cases.
- Provide a `get_basic_fields` class method listing the most relevant fields

## ⚠️ Python Environment (IMPORTANT)

**Local Development:** Use the conda `nodetool` environment. Do not use system Python.

```bash
conda activate nodetool
# then run commands normally
```

**GitHub CI / Copilot Agent:** Uses standard Python 3.11 with pip. Dependencies are pre-installed via `.github/workflows/copilot-setup-steps.yml`. Run commands directly without conda.

## Commands

After adding or changing nodes run this command to regenerate
`src/nodetool/package_metadata/nodetool-huggingface.json`. CI fails when the
committed file is out of date.

```bash
uv run nodetool-pkg scan --write
```

## Linting and Tests

Before submitting a pull request, run the following checks:

```bash
uv lock --check
uv run ruff check .
uv run pytest -q
```

After changing dependencies in `pyproject.toml`, run `uv lock` and commit
`uv.lock`.

A release bump changes the version in `pyproject.toml`, `ARG HF_VERSION` in
`Dockerfile` and both `HF_VERSION` values in `docker-compose.yaml`, then runs
`uv lock` and `uv run nodetool-pkg scan --write` and commits the results. The
publish workflow refuses a tag whose files disagree.

Formatting issues or lint errors should be fixed before committing. Test coverage is expected to be added when applicable.
