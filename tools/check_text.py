#!/usr/bin/env python3

# The MIT License (MIT)
#
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Check and normalize repository-wide text-file whitespace conventions."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Sequence


def _strip_trailing_whitespace(line: bytes, preserve_markdown_breaks: bool) -> bytes:
    content = line.rstrip(b" \t")
    whitespace = line[len(content) :]
    if preserve_markdown_breaks and content and whitespace == b"  ":
        return line
    return content


def normalize_content(content: bytes, path: Path) -> bytes:
    """Return content with LF endings, clean line ends, and a final newline."""

    normalized = content.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    preserve_markdown_breaks = path.suffix.lower() in {".md", ".markdown"}
    normalized = b"\n".join(
        _strip_trailing_whitespace(line, preserve_markdown_breaks)
        for line in normalized.split(b"\n")
    )
    if normalized and not normalized.endswith(b"\n"):
        normalized += b"\n"
    return normalized


def describe_issues(content: bytes, normalized: bytes, path: Path) -> list[str]:
    """Describe the conventions that differ between content and normalized."""

    issues = []
    if b"\r" in content:
        issues.append("non-LF line ending")

    preserve_markdown_breaks = path.suffix.lower() in {".md", ".markdown"}
    for line in content.replace(b"\r\n", b"\n").replace(b"\r", b"\n").split(b"\n"):
        if _strip_trailing_whitespace(line, preserve_markdown_breaks) != line:
            issues.append("trailing whitespace")
            break

    if content and not content.endswith((b"\n", b"\r")):
        issues.append("missing final newline")
    if content != normalized and not issues:
        issues.append("text formatting")
    return issues


def process_files(mode: str, filenames: Iterable[str]) -> int:
    """Check or fix filenames and return a process-style status code."""

    failed = False
    for filename in filenames:
        path = Path(filename)
        content = path.read_bytes()
        normalized = normalize_content(content, path)
        if content == normalized:
            continue

        if mode == "fix":
            path.write_bytes(normalized)
            print(f"fixed {path}")
            continue

        failed = True
        issues = ", ".join(describe_issues(content, normalized, path))
        print(f"{path}: {issues}")

    return 1 if failed else 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("check", "fix"))
    parser.add_argument("filenames", nargs="+")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    return process_files(args.mode, args.filenames)


if __name__ == "__main__":
    raise SystemExit(main())
