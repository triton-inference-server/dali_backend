#!/bin/sh
set -eu

test_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
# Use the test server stub and the installed wheel after the build tree is removed.
export LD_LIBRARY_PATH="${test_dir}:${test_dir}/../../backends/dali/wheel/dali/nvidia/dali${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
exec "${test_dir}/unittests" "$@"
