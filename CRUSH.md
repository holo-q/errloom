# CRUSH.md

Build/run
- uv sync --extra all; uv pip install flash-attn --no-build-isolation
- uv run main                      # CLI entry; see AGENT.md for modes
- uv run main --dry --n 1 --debug  # Mock client dry run
- uv run main <ware|loom> <dry|train|dump> [n] [opts]

Test
- uv run testslide errloom/test_holoware_parse_testslide.py  # pre-commit hook path
- uv run testslide tests/                                    # run all TestSlide tests
- uv run pytest tests -q                                     # if pytest-based tests are added
- Single test: uv run pytest tests/test_holoware.py::test_basic -q

Docs
- (Sphinx in docs/) make -C docs html

Lint/typecheck
- ruff/mypy not configured; if present use:
  - uv run ruff check errloom
  - uv run mypy errloom

Style
- Python 3.11–3.12, run via uv. No comments unless explicitly requested. Avoid kwargs unless shaping a public API. Prefer mutable structs/classes over functional patterns. Strong typing; explicit imports from existing modules only; never assume libs not in pyproject. Security: never log secrets; no plaintext keys in code. Logging via errloom.lib.log (RichHandler, color utilities); prefer logger from getLogger/setup_logging; keep console concise, file logs detailed. Errors: raise explicit exceptions; in CLI paths, show_help on invalid args; use log_design ethos from CLAUDE/AGENT. Naming: snake_case for vars/functions, PascalCase for classes, UPPER_SNAKE for consts. Imports: absolute within package (from errloom.*) following existing patterns. Formatting: black-like (120 width common in logs), keep lines readable. CLI: favor uv run main, flags in argp.py. Testing: prefer TestSlide where applicable; mocks over network calls.

Cursor/Copilot rules
- Include .cursor/rules/base.mdc guidance (coding standard, logging, commands). Follow Pair Programming and Work Standards from CLAUDE.md/AGENT.md.

Notes
- Project scripts: [project.scripts] main/errl in pyproject.toml. VLLM/flash-attn heavy deps; uv toolchain recommended. Use runs/, logs/ directories created by code.
