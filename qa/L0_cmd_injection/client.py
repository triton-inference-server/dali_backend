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

import tritonclient.grpc as t_client
from tritonclient.utils import InferenceServerException
import argparse

def assert_error(func, *args, contains=None):
  try:
    func(*args)
    msg = "Should raise error: " + ', '.join([str(arg) for arg in args])
    assert False, msg
  except InferenceServerException as err:
    err_msg = str(err)
  if contains is not None:
    assert contains in err_msg, f'Error message:\n  {err_msg}\nshould contain:\n{contains}'

def test_loading(url):
  client = t_client.InferenceServerClient(url=url)

  assert_error(client.load_model, 'dali_cmd_injection',
               contains="Invalid character found in model path. The path contains a forbidden character: '''")
  print('dali_cmd_injection test OK')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-u', '--url', type=str, required=False, default='localhost:8001',
                        help='Inference server GRPC URL. Default is localhost:8001.')
    return parser.parse_args()


if __name__ == '__main__':
  args = parse_args()
  test_loading(args.url) 