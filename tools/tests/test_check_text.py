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

import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from tools.check_text import normalize_content, process_files


class NormalizeContentTest(unittest.TestCase):
    def test_normalizes_line_endings_whitespace_and_final_newline(self):
        path = Path("example.txt")
        content = b"first  \r\nsecond\t\rthird"

        self.assertEqual(normalize_content(content, path), b"first\nsecond\nthird\n")

    def test_preserves_markdown_hard_line_break(self):
        path = Path("example.md")
        content = b"hard break  \nblank   \n"

        self.assertEqual(normalize_content(content, path), b"hard break  \nblank\n")

    def test_preserves_empty_content(self):
        self.assertEqual(normalize_content(b"", Path("empty.txt")), b"")


class ProcessFilesTest(unittest.TestCase):
    def test_check_reports_without_modifying_and_fix_normalizes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "example.txt"
            path.write_bytes(b"value ")

            with contextlib.redirect_stdout(io.StringIO()):
                check_status = process_files("check", [str(path)])
            self.assertEqual(check_status, 1)
            self.assertEqual(path.read_bytes(), b"value ")
            with contextlib.redirect_stdout(io.StringIO()):
                fix_status = process_files("fix", [str(path)])
            self.assertEqual(fix_status, 0)
            self.assertEqual(path.read_bytes(), b"value\n")


if __name__ == "__main__":
    unittest.main()
