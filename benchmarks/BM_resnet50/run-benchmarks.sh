#!/bin/bash -ex

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

DATASETS_DIR=${DATASETS_DIR:-'/data/imagenet/ILSVRC2012_rawdata'}

echo "Benchmark dataset preprocessing"
python scripts/dataset_preprocess.py --perf-file results/preprocessing.json

echo "LOAD MODELS"
python scripts/model-loader.py load -m dali
python scripts/model-loader.py load -m resnet50_trt
python scripts/model-loader.py load -m dali_trt_resnet50

TIME_WINDOW=10000
BATCH_SIZES="2 8 16 32 64 128"

echo "WARM-UP"
perf_analyzer -m dali_trt_resnet50 --input-data imagenet64.json --concurrency-range=128 -p$TIME_WINDOW

echo "NN Benchmarks: single-sample"
perf_analyzer -m resnet50_trt -p$TIME_WINDOW --concurrency-range=16:128:16

echo "NN Benchmarks: batched"
for BS in $BATCH_SIZES ; do
  perf_analyzer -m resnet50_trt -p$TIME_WINDOW -b$BS ;
done

echo "Ensemble Benchmarks: single-sample"
perf_analyzer -m dali_trt_resnet50 -p$TIME_WINDOW --input-data imagenet64.json --concurrency-range=16:128:16

echo "Ensemble Benchmarks: batched"
for BS in $BATCH_SIZES ; do
  perf_analyzer -m dali_trt_resnet50 -p$TIME_WINDOW --input-data inputs-data/ --shape input:`stat --printf="%s" inputs-data/input` -b$BS ;
done
