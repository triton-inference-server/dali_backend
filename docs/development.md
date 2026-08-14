# Development checks

The repository uses [pre-commit](https://pre-commit.com/) as a versioned runner for Python linting
and Python/C++ formatting. Install the pinned runner in a virtual environment:

```shell
python3.10 -m venv .venv
. .venv/bin/activate
python -m pip install --requirement requirements-lint.txt
pre-commit install --hook-type pre-commit --hook-type pre-push
```

The first run downloads the tool environments pinned in `.pre-commit-config.yaml`.

## Check and fix commands

Checks never edit files. Check staged files or every file changed by the current branch:

```shell
pre-commit run
pre-commit run --from-ref origin/main --to-ref HEAD
```

Formatting fixes use the dedicated manual hook stage:

```shell
pre-commit run --hook-stage manual
pre-commit run --hook-stage manual --from-ref origin/main --to-ref HEAD
```

After applying fixes, inspect and stage the result, then rerun the corresponding check command.
Use `pre-commit run --all-files` only to audit the legacy formatting backlog. Normal development
and CI intentionally check changed files so introducing the setup does not rewrite unrelated code.

## Tool and file scope

| Files | Checks | Pinned version |
| --- | --- | --- |
| `src/**/*.cc`, `src/**/*.h` | clang-format using `.clang-format` | 16.0.6 |
| `**/*.py` | Ruff formatting and high-signal syntax/name checks | 0.16.0 |

Ruff replaces separate Black, isort, and Flake8 installations. Its initial lint rule set focuses on
syntax errors and undefined names; enabling broader style rules would require unrelated cleanup.
clang-format checks layout but does not perform semantic C++ linting. clang-tidy is not part of the
commit hook because it needs a complete CUDA, DALI, and Triton compile database; compiler warnings
remain part of the normal build and clang-tidy can be added to a suitable CI build separately.

## Exclusions

The following paths are excluded from all hooks:

- `extern/`, which contains the Catch2 submodule.
- `benchmarks/dali_vs_python/BM_jasper/model_repository/jasper_python/1/features.py`, a snapshot
  from NVIDIA Deep Learning Examples documented by the benchmark README.
- `src/utils/cmake_config.h`, which CMake generates from `cmake_config.h.in`.

The hooks are filtered by file type, so Python checks run only for Python files and clang-format
runs only for C++ source and headers under `src/`.

## Continuous integration

The GitHub Actions lint workflow uses the same pinned pre-commit configuration and checks the
changed range for pull requests and pushes to `main`. The separate internal GitLab CI repository
does not consume GitHub workflow files; adding the same gate there requires an independently
authorized CI-repository change.
