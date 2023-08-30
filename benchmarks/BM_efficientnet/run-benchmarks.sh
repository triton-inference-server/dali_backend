# The MIT License (MIT)
#
# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES
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

: ${MAX_BATCH_SIZE:=${1:-64}}

load_models() {
  echo "Loading models..."
  python scripts/model-loader.py -u "${GRPC_ADDR}" load -m preprocessing
  python scripts/model-loader.py -u "${GRPC_ADDR}" load -m efficientnet-b0
  python scripts/model-loader.py -u "${GRPC_ADDR}" load -m efficientnet_ensemble
  sleep 5
  echo "...models loaded"
}

unload_models() {
  echo "Unloading models..."
  python scripts/model-loader.py -u "${GRPC_ADDR}" unload -m preprocessing
  python scripts/model-loader.py -u "${GRPC_ADDR}" unload -m efficientnet-b0
  python scripts/model-loader.py -u "${GRPC_ADDR}" unload -m efficientnet_ensemble
  sleep 5
  echo "...models unloaded"
}

# The BATCH_SIZES are assigned with powers of 2 lower or equal to MAX_BATCH_SIZE
BATCH_SIZES=()
POWER=0
while ((2**POWER <= MAX_BATCH_SIZE)); do
    BATCH_SIZES+=("$((2**POWER))")
    ((POWER++))
done

# Feel free to assign BATCH_SIZES manually...
# BATCH_SIZES="1 2 4 8 16 32 64"

CONCURRENCY_RANGE="16:512:16"
GRPC_ADDR="localhost:8001"
TIME_WINDOW=10000
PERF_ANALYZER_ARGS="-i grpc -u $GRPC_ADDR -p$TIME_WINDOW --verbose-csv --collect-metrics"
INPUT_NAME="INPUT"
BENCH_DIR="bench-$(date +%Y%m%d_%H%M%S)"

mkdir -p "$BENCH_DIR"

for BS in $BATCH_SIZES; do
  echo "Efficientnet Benchmark. Batch size: $BS"
  load_models
  perf_analyzer $PERF_ANALYZER_ARGS -m efficientnet_ensemble --input-data test_sample --shape $INPUT_NAME:$(stat --printf="%s" test_sample/$INPUT_NAME) --concurrency-range=$CONCURRENCY_RANGE -b "$BS" -f "$BENCH_DIR/report-$BS.csv"
  unload_models
done

pip install -U pandas
python scripts/concatenate_results.py -p "$BENCH_DIR"
