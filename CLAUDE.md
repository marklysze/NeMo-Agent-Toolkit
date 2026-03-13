# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NVIDIA NeMo Agent Toolkit — an enterprise-grade Python toolkit that adds intelligence to AI agents across any framework through instrumentation, observability, and continuous learning. Package name is `nvidia-nat`, CLI/API namespace is `nat`, env var prefix is `NAT_`.

## Build & Development

**Prerequisites:** Python >=3.11,<3.14, uv 0.9.28

```bash
# Install all packages in development mode
uv sync --all-extras

# Install specific extras
uv sync --extra langchain --extra eval
```

## Common Commands

```bash
# Linting and formatting (run before pushing)
pre-commit run --all-files

# Lint only
ruff check --fix .

# Format only
yapf -i -r --style ./pyproject.toml .

# Run tests for a specific package
pytest packages/nvidia_nat_core
pytest packages/nvidia_nat_langchain

# Run a single test file or test
pytest packages/nvidia_nat_core/tests/test_something.py
pytest packages/nvidia_nat_core/tests/test_something.py::test_specific_function

# Include slow/integration tests (excluded by default)
pytest --run_slow packages/nvidia_nat_core
pytest --run_integration packages/nvidia_nat_core
pytest --run_slow --run_integration packages/nvidia_nat_core

# Local CI simulation
ci/scripts/run_ci_local.sh checks   # style, format, copyright
ci/scripts/run_ci_local.sh tests    # full test suite
ci/scripts/run_ci_local.sh all      # everything
```

## Architecture

**Monorepo with 31 packages** in `packages/`, each with its own `pyproject.toml`, `src/`, `tests/`, and `uv.lock`. All are editable installs wired through the root `pyproject.toml` via `[tool.uv.sources]`.

**Core package** (`packages/nvidia_nat_core/src/nat/`):
- `agents/` — Agent implementations (ReAct, ReWOO, Tool Calling, custom)
- `cli/` — `nat` command-line interface
- `data_models/` — Pydantic models for YAML-driven workflow configuration
- `functions/` — Function registry and base classes (building blocks of workflows)
- `llms/` — Pluggable LLM interface (NIM, OpenAI, Bedrock, Azure)
- `front_ends/` — FastAPI server, streaming, async jobs
- `plugins/` — Plugin framework (eval, observability)
- `workflows/` — Workflow orchestration engine
- `builder/` — Runtime construction from config (Builder pattern, context state)

**Framework integration packages** wrap external agent frameworks (LangChain, LlamaIndex, CrewAI, ADK, Strands, AutoGen, Agno, Semantic Kernel) to add toolkit capabilities.

**Plugin packages** add optional features: MCP/A2A protocol support, observability (OpenTelemetry, Phoenix, Weave), evaluation, data flywheel, memory backends (Redis, Zep, Mem0).

**Dependency conflicts:** `openpipe-art` and `ragaai` conflict with several other extras (see `[tool.uv]` conflicts in `pyproject.toml`). The `most` extra includes everything except these two.

## Coding Standards

- **Formatting:** yapf (PEP 8 base, 120-char line limit), then ruff for linting. 4-space indents.
- **Linting rules:** E, F, W (flake8), I (isort), PL (pylint subset), UP (pyupgrade). Config in `pyproject.toml`.
- **Type hints** required on all public API parameters and return values (Python 3.11+ syntax).
- **Docstrings:** Google-style for all public modules, classes, functions.
- **SPDX headers:** Every source file must start with the Apache-2.0 SPDX header (copy from an existing file).
- **Exception handling:** Use bare `raise` to preserve stack traces; `logger.error()` when re-raising, `logger.exception()` when not re-raising.
- **Pydantic SecretStr bug:** Never use `default=""` with `SecretStr`/`SerializableSecretStr`/`OptionalSecretStr`. Use `default=None` for optional or `default_factory=lambda: SerializableSecretStr("")` for required.
- **Dependencies:** Use `~=<version>` with two-digit versions. Add to `pyproject.toml` alphabetically and update `uv.lock`.

## Testing Conventions

- **Framework:** pytest + pytest-asyncio (async mode: auto, session-scoped event loop)
- **Global timeout:** 300s per test (override with `@pytest.mark.timeout(seconds)`)
- **Markers:** `@pytest.mark.slow` (>30s), `@pytest.mark.integration` (needs external services), `@pytest.mark.benchmark`
- **Fixture naming:** Decorator gets `name="my_fixture"`, function is `def fixture_my_fixture():`
- **Test LLM:** Use `nat_test_llm` from `nvidia-nat-test` for deterministic, API-key-free testing
- **Coverage target:** >=80%
- **Examples are ignored** by default in pytest (`--ignore=examples/`). Also ignored: `nvidia_nat_openpipe_art`, `nvidia_nat_ragaai`.

## Naming & Terminology

- **Full name (first use in docs):** "NVIDIA NeMo Agent Toolkit"
- **Short name (subsequent):** "NeMo Agent Toolkit" or "the toolkit"
- **Never use** in documentation: "NAT", "Agent Intelligence toolkit", "AgentIQ", "aiq" (deprecated names)
- **Code identifiers:** `nat` (API/CLI namespace), `nvidia-nat` (PyPI package), `NAT_` (env vars)
- Do not modify `CHANGELOG.md` content
