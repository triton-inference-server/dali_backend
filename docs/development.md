# Development checks

The repository uses [pre-commit](https://pre-commit.com/) as a versioned runner for formatting
and linting. Install the pinned runner in a virtual environment:

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

The text formatter can be tested independently with:

```shell
python -m unittest discover -s tools/tests -v
```

## Tool and file scope

| Files | Checks | Pinned version |
| --- | --- | --- |
| `src/**/*.cc`, `src/**/*.h` | clang-format using `.clang-format` | 16.0.6 |
| `**/*.py` | Ruff formatting and high-signal syntax/name checks | 0.16.0 |
| `CMakeLists.txt`, `*.cmake`, `*.cmake.in` | cmake-format, 80-column width | 0.6.13 |
| `**/*.sh` | shfmt with two-space/case indentation; ShellCheck errors under Bash semantics | 3.13.1 / 0.11.0 |
| `*.yaml`, `*.yml`, `*.toml` | syntax validation | pre-commit-hooks 6.0.0 |
| Text files, including Markdown and configuration | LF endings, trailing whitespace, final newline | repository script |

Ruff replaces separate Black, isort, and Flake8 installations. Its initial lint rule set focuses on
syntax errors and undefined names; enabling broader style rules would require unrelated cleanup.
clang-tidy is not part of this fast check because it needs a complete CUDA, DALI, and Triton compile
database; compiler warnings remain part of the normal build.

`.cmake-format.py` uses cmake-format's configuration DSL and is not ordinary Python source, so Ruff
does not inspect it.

Markdown is not reflowed, and two-space Markdown hard line breaks are preserved. The YAML syntax
hook excludes `cmake/dalienv.yml.in`, whose CMake substitution can insert a YAML fragment.

## Exclusions

The following paths are excluded from all hooks:

- `extern/`, which contains the Catch2 submodule.
- `benchmarks/dali_vs_python/BM_jasper/model_repository/jasper_python/1/features.py`, a snapshot
  from NVIDIA Deep Learning Examples documented by the benchmark README.
- `docs/examples/efficientnet/0001-Update-requirements-and-add-Dockerfile.bench.patch`, where
  whitespace is part of the patch payload.
- `src/utils/cmake_config.h`, which CMake generates from `cmake_config.h.in`.

Binary media and serialized data are skipped by pre-commit's file typing. Protobuf text,
Dockerfiles, documentation, and other text without a language-specific formatter still receive
the line-ending and whitespace checks.

## Continuous integration

The GitHub Actions lint workflow uses the same pinned pre-commit configuration and checks the
changed range for pull requests and pushes to `main`. The separate internal GitLab CI repository
does not consume GitHub workflow files; adding the same gate there requires an independently
authorized CI-repository change.
