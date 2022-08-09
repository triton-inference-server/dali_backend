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

import tritonclient.grpc as t_client
import argparse


def check_config(config, bs, out_names):
  assert config['max_batch_size'] == bs

  inps = config['input']
  assert len(inps) == 2
  assert inps[0]['name'] == 'DALI_INPUT_0'
  assert inps[0]['data_type'] == 'TYPE_FP16'
  assert inps[0]['dims'] == ['-1']
  assert inps[0]['allow_ragged_batch'] == True
  assert inps[1]['name'] == 'DALI_INPUT_1'
  assert inps[1]['data_type'] == 'TYPE_FP16'
  assert inps[1]['dims'] == ['-1']
  assert inps[1]['allow_ragged_batch'] == True

  outs = config['output']
  assert len(outs) == 2
  assert outs[0]['name'] == out_names[0]
  assert outs[0]['data_type'] == 'TYPE_FP16'
  assert outs[0]['dims'] == ['-1']
  assert outs[1]['name'] == out_names[1]
  assert outs[1]['data_type'] == 'TYPE_FP32'
  assert outs[1]['dims'] == ['-1']


def test_configs(url):
  client = t_client.InferenceServerClient(url=url)

  conf1 = client.get_model_config("full_autoconfig", as_json=True)
  check_config(conf1['config'], 256, ['__ArithmeticGenericOp_2', '__ArithmeticGenericOp_4'])

  conf2 = client.get_model_config("partial_autoconfig", as_json=True)
  check_config(conf2['config'], 32, ['DALI_OUTPUT_0', 'DALI_OUTPUT_1'])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-u', '--url', type=str, required=False, default='localhost:8001',
                        help='Inference server GRPC URL. Default is localhost:8001.')
    return parser.parse_args()


if __name__ == '__main__':
  args = parse_args()
  test_configs(args.url)
