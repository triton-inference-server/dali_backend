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

import tensorrt as trt
import argparse


def build_engine(logger, onnx_path, input_name, batch_size_info, sample_shape = [224, 224, 3]):

   """
   This is the function to create the TensorRT engine
   Args:
      onnx_path : Path to onnx_file. 
      shape : Shape of the input of the ONNX file. 
  """
   with trt.Builder(logger) as builder, builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, trt.OnnxParser(network, logger) as parser:
    max_workspace_size = (256 << 20)
    with open(onnx_path, 'rb') as model:
        parser.parse(model.read())
    shape_info = {
      'min': (batch_size_info['min'], *sample_shape),
      'opt': (batch_size_info['opt'], *sample_shape),
      'max': (batch_size_info['max'], *sample_shape)
    }
    config = builder.create_builder_config()
  #  config.flags |= bool(fp16_mode) << int(trt.BuilderFlag.FP16)
    config.max_workspace_size = max_workspace_size
    profile = builder.create_optimization_profile()
    profile.set_shape(input_name, **shape_info)
    config.add_optimization_profile(profile)
    engine = builder.build_engine(network, config=config)
    return engine

def save_engine(engine, file_name):
  buf = engine.serialize()
  with open(file_name, 'wb') as f:
    f.write(buf)


def load_engine(trt_runtime, plan_path):
  with open(plan_path, 'rb') as f:
    engine_data = f.read()
  engine = trt_runtime.deserialize_cuda_engine(engine_data)
  return engine


def get_args():
  parser = argparse.ArgumentParser(description='Convert an ONNX model to a TensorRT engine.')
  parser.add_argument('-f', '--onnx_file', required=True, action='store', help='ONNX model file.')
  parser.add_argument('-b', '--batch_size', required=False, action='store', default=128,
                      help='Max batch size of the model.')
  parser.add_argument('-o', '--output',  required=True, action='store', help='Name of the output TensorRT engine file.')
  return parser.parse_args()

def main(args):
  TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
  trt_runtime = trt.Runtime(TRT_LOGGER)
  engine = build_engine(TRT_LOGGER, args.onnx_file, 'input_tensor:0', {'min': 1, 'opt': 64, 'max': args.batch_size})
  save_engine(engine, args.output)

if __name__ == '__main__':
  args = get_args()
  main(args)
