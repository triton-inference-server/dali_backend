#!/bin/bash -e

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

set -e

CHECKPOINT_DIR=${CHECKPOINT_DIR:-'checkpoints'}

source scripts/download_checkpoint.sh

# Setup tensorflow model
mkdir -p model_repository/resnet50_tf/1
cp -r ${CHECKPOINT_DIR}/saved_model/nvidia_rn50_tf_amp model_repository/resnet50_tf/1/model.savedmodel
echo "Succesfully set up tensorflow model"

# Setup ONNX model
mkdir -p model_repository/resnet50_onnx/1
python3 -m tf2onnx.convert --saved-model ${CHECKPOINT_DIR}/saved_model/nvidia_rn50_tf_amp/ --output model_repository/resnet50_onnx/1/model.onnx
echo "Succesfully set up ONNX model"

# Setup TensorRT model
mkdir -p model_repository/resnet50_trt/1
# python3 scripts/onnx2trt.py -f model_repository/resnet50_onnx/1/model.onnx -o model_repository/resnet50_trt/1/model.plan
trtexec --onnx=model_repository/resnet50_onnx/1/model.onnx --saveEngine=model_repository/resnet50_trt/1/model.plan  --explicitBatch --minShapes=\'input\':1x224x224x3 --optShapes=\'input\':64x224x224x3 --maxShapes=\'input\':128x224x224x3
echo "Succesfully set up TensorRT model"

# Setup DALI preprocessing model
mkdir -p model_repository/dali/1
python3 model_repository/dali/pipeline.py model_repository/dali/1/model.dali
echo "Succesfully set up DALI preprocessing model"

# Setup DALI + TRT ensemble
mkdir -p model_repository/dali_trt_resnet50/1
echo "Successfully set up DALI + TensorRT ensemble"
