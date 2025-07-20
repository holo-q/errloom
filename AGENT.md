# AGENT.md - Thauten Development Guide

## Commands
- **Run dry test**: `uv run train_compressor.py --dry --n 1 --log-more` (MockClient returns MOCK instead of calling vllm)
- **Debug**: Add `--debug` flag to see debug log level
- **Test**: `uv run pytest -s` (run before all commits)
- **Run single test**: `uv run pytest -s path/to/test_file.py::test_function_name`
- **Build/Check**: No explicit build command (Python project with uv)

## Architecture
- **Main packages**: `thauten/` (core), `errloom/` (RL framework), `hol/` (holoware files)
- **Key files**: `train_compressor.py` (compression training), `errloom/main.py` (main execution)
- **Holoware**: `.hol` files contain prompting programs with cognitive fences (`<compress>`, `<think>`, etc.)
- **Training**: GRPO-based RL for symbolic compression and reasoning development
- **Tools**: Uses rich for logging, torch/transformers for ML, pydantic for data validation

## Code Style (.cursor/rules/base.mdc)
- **No kwargs** unless serious API design interest
- **Anti-functional**: Write Rust/C#-style with mutable structs, avoid functional patterns
- **Logging**: Use logging + rich with colors, structure, minimal verbosity, maintain `log_design.md`
- **Imports**: Follow existing patterns, use uv for dependency management
- **Commit**: Always check `git status` first, never commit without asking
- **Proactive**: Ask questions, propose next steps, enumerate ambiguities before implementation
