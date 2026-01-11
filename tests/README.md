# Tests for nodetool-huggingface

This directory contains the test suite for the nodetool-huggingface package.

## Test Structure

- `conftest.py` - Pytest configuration and shared fixtures
- `unit/` - Unit tests that test individual components in isolation
  - `test_node_imports.py` - Tests that all node modules can be imported
  - `test_nodes.py` - Tests for node creation and basic functionality
  - `test_utils.py` - Tests for utility functions
- `integration/` - Integration tests that test components working together
  - `test_dsl_nodes.py` - Tests for DSL node instantiation

## Running Tests

### Install Dependencies

```bash
pip install -e ".[test]"
```

### Run All Tests

```bash
pytest
```

### Run Unit Tests Only

```bash
pytest -m unit
```

### Run Integration Tests Only

```bash
pytest -m integration
```

### Run Tests with Coverage

```bash
pytest --cov=src/nodetool --cov-report=html
```

### Run Specific Test File

```bash
pytest tests/unit/test_nodes.py -v
```

## Test Markers

Tests are marked with pytest markers to allow selective execution:

- `@pytest.mark.unit` - Unit tests
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.slow` - Slow tests
- `@pytest.mark.requires_gpu` - Tests requiring GPU
- `@pytest.mark.requires_model` - Tests requiring model downloads

## CI/CD

Tests run automatically on GitHub Actions via the `.github/workflows/test.yml` workflow file.

The workflow:
1. Installs nodetool-core from GitHub main branch
2. Installs the package with test dependencies
3. Runs unit tests
4. Runs integration tests
5. Generates coverage reports

## Writing Tests

When adding new nodes or functionality:

1. Add unit tests in `tests/unit/` to test the component in isolation
2. Add integration tests in `tests/integration/` if the component interacts with other parts
3. Use appropriate markers (`@pytest.mark.unit`, `@pytest.mark.integration`)
4. Mock external dependencies and model loading to keep tests fast
5. Use fixtures from `conftest.py` for common test data

## Test Coverage

Current test coverage includes:
- Node module imports (34 tests)
- Node instantiation and basic functionality (23 tests)
- Utility functions (11 tests)
- DSL node creation (7 tests)

Total: 41 tests
