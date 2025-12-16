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

After adding or changing nodes run these commands to generate metadata and DSL.

```bash
nodetool package scan
nodetool codegen
```

## Linting and Tests

Before submitting a pull request, run the following checks:

```bash
ruff check .
black --check .
pytest -q
```

Formatting issues or lint errors should be fixed before committing. Test coverage is expected to be added when applicable.
