#!/bin/bash -e

# The MIT License (MIT)
#
# Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES
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

mkdir -p model_repository/model0_gpu_valid/1
cp pipeline_gpu.py model_repository/model0_gpu_valid/1/dali.py
echo "model0_gpu_valid ready"

mkdir -p model_repository/model1_gpu_invalid_i_type/1
cp pipeline_gpu.py model_repository/model1_gpu_invalid_i_type/1/dali.py
echo "model1_gpu_invalid_i_type ready"

mkdir -p model_repository/model2_gpu_invalid_o_ndim/1
cp pipeline_gpu.py model_repository/model2_gpu_invalid_o_ndim/1/dali.py
echo "model2_gpu_invalid_o_ndim ready"

mkdir -p model_repository/model3_cpu_valid/1
cp pipeline_cpu.py model_repository/model3_cpu_valid/1/dali.py
echo "model3_cpu_valid ready"

mkdir -p model_repository/model4_cpu_invalid_missing_output/1
cp pipeline_cpu.py model_repository/model4_cpu_invalid_missing_output/1/dali.py
echo "model4_cpu_invalid_missing_output ready"
