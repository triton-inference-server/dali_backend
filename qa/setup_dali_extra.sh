#!/bin/bash

# The MIT License (MIT)
#
# Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES
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
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


set -ex

# Fetch test data
export DALI_EXTRA_PATH=${DALI_EXTRA_PATH:-/opt/dali_extra}
export DALI_EXTRA_URL=${DALI_EXTRA_URL:-"https://github.com/NVIDIA/DALI_extra.git"}
export DALI_EXTRA_NO_DOWNLOAD=${DALI_EXTRA_NO_DOWNLOAD}

DIR=$(cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )
DALI_EXTRA_VERSION_PATH="${DALI_BACKEND_REPO_ROOT}/DALI_EXTRA_VERSION"
DALI_EXTRA_VERSION=${DALI_EXTRA_VERSION_SHA:-$(cat ${DALI_EXTRA_VERSION_PATH})}
echo "Using DALI_EXTRA_VERSION = ${DALI_EXTRA_VERSION}"
if [ ! -d "$DALI_EXTRA_PATH" ] && [ "${DALI_EXTRA_NO_DOWNLOAD}" == "" ]; then
    git clone "$DALI_EXTRA_URL" "$DALI_EXTRA_PATH"
fi

pushd "$DALI_EXTRA_PATH"
if [ "${DALI_EXTRA_NO_DOWNLOAD}" == "" ]; then
    git fetch origin ${DALI_EXTRA_VERSION}
fi
git checkout ${DALI_EXTRA_VERSION}
popd
