#!/bin/bash

# The MIT License (MIT)
#
# Copyright (c) 2024 NVIDIA CORPORATION
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

if [ -z $1 ]; then
    echo "
Script that updates base Triton and DLFW containers versions in the DALI Backend repository.
IMPORTANT: run this script from the repository root directory.
Usage: ./tools/update_triton_version.sh [new container version]
  e.g: ./tools/update_triton_version.sh 24.03
"
    exit 1
fi

is_correct_version_re='^[0-9][0-9]\.[0-9][0-9]$'
if ! [[ $1 =~ $is_correct_version_re ]]; then
    echo "Provided version ($1) does not look correct. Please double-check."
    exit 1
fi

echo "Updating the version to $1"

# Dockerfile.release
sed -i "s/ARG TRITON_VERSION=24.02/ARG TRITON_VERSION=$1/g" docker/Dockerfile.release

# Dockerfile.devel
sed -i "s/ARG TRITON_VERSION=24.02/ARG TRITON_VERSION=$1/g" docker/Dockerfile.devel

# Efficientnet example
sed -i "s|+ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:24.02-py3|+ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:$1-py3|g" docs/examples/efficientnet/0001-Update-requirements-and-add-Dockerfile.bench.patch
sed -i "s|FROM nvcr.io/nvidia/tritonserver:24.02-py3|FROM nvcr.io/nvidia/tritonserver:$1-py3|g" benchmarks/BM_efficientnet/Dockerfile.torch
