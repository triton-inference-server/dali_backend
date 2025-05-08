#!/bin/bash -ex

# The MIT License (MIT)
#
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
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

cp -r ${DALI_BACKEND_REPO_ROOT}/docs/examples/resnet50_trt/model_repository/* model_repository/
cp -r ${DALI_BACKEND_REPO_ROOT}/docs/examples/resnet50_trt/onnx_exporter.py .
cp -r ${DALI_BACKEND_REPO_ROOT}/docs/examples/resnet50_trt/serialize_dali_pipeline.py .

cp model_repository/resnet50_trt/labels.txt model_repository/resnet50_onnx/
rm -r model_repository/resnet50_trt/

mkdir -p model_repository/dali/1
mkdir -p model_repository/ensemble_dali_resnet50/1
mkdir -p model_repository/resnet50_onnx/1


python3 onnx_exporter.py --save ./model_repository/resnet50_onnx/1/model.onnx
python3 serialize_dali_pipeline.py --save ./model_repository/dali/1/model.dali

sed -i -e 's/resnet50_trt/resnet50_onnx/g' ./model_repository/ensemble_dali_resnet50/config.pbtxt
