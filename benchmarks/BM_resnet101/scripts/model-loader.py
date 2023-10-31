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

import tritonclient.grpc as t_client
import argparse
import sys

def get_args():
  parser = argparse.ArgumentParser(description='Load or unload a model in Triton server.')
  parser.add_argument('action', action='store', choices=['load', 'unload', 'reload'])
  parser.add_argument('-u', '--url', required=False, action='store', default='localhost:8001', help='Server url.')
  parser.add_argument('-m', '--model', required=True, action='store', help='Model name.')
  return parser.parse_args()


def main(args):
  client = t_client.InferenceServerClient(url=args.url)
  if args.action in ['reload', 'unload']:
    client.unload_model(args.model)
    print('Successfully unloaded model', args.model)

  if args.action in ['reload', 'load']:
    client.load_model(args.model)
    print('Successfully loaded model', args.model)


if __name__ == '__main__':
  args = get_args()
  main(args)
