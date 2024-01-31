#!/bin/bash -ex

# The MIT License (MIT)
#
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES
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

load_models() {
  echo "Loading models..."
  python scripts/model-loader.py -u "${GRPC_ADDR}" load -m dali
  sleep 5
  echo "...models loaded"
}

unload_models() {
  echo "Unloading models..."
  python scripts/model-loader.py -u "${GRPC_ADDR}" unload -m dali
  sleep 5
  echo "...models unloaded"
}

GRPC_ADDR=${GRPC_ADDR:-"localhost:8001"}
TIME_WINDOW=10000
PERF_ANALYZER_ARGS="-i grpc -u $GRPC_ADDR -p$TIME_WINDOW --verbose-csv --collect-metrics"
INPUT_NAME="DALI_INPUT_0"

nvidia-smi -q -i 0 -x > /tmp/mu_pre.xml

load_models
perf_analyzer $PERF_ANALYZER_ARGS -m dali --input-data test_sample --shape $INPUT_NAME:$(stat --printf="%s" test_sample/$INPUT_NAME) -b 64
unload_models

nvidia-smi -q -i 0 -x > /tmp/mu_post.xml

python scripts/compare_memory_usage.py
