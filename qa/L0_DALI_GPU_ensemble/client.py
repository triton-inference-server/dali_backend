# The MIT License (MIT)
#
# Copyright (c) 2021 NVIDIA CORPORATION
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

import numpy as np
from dali_backend.test_utils.client import TestClient
from numpy.random import randint, random
import argparse

# TODO: Use actual DALI pipelines to calculate ground truth
def ref_func(inp1, inp2):
  return inp1 * 2 / 3, (inp2 * 3).astype(np.half).astype(np.single) / 2

def random_gen(max_batch_size):
  while True:
    size1 = randint(100, 300)
    size2 = randint(100, 300)
    bs = randint(1, max_batch_size + 1)
    yield random((bs, size1)).astype(np.single), \
          random((bs, size2)).astype(np.single)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-u', '--url', type=str, required=False, default='localhost:8001',
                        help='Inference server GRPC URL. Default is localhost:8001.')
    parser.add_argument('-n', '--n_iters', type=int, required=False, default=1, help='Number of iterations')
    parser.add_argument('-c', '--concurrency', type=int, required=False, default=1,
                        help='Request concurrency level')
    parser.add_argument('-b', '--max_batch_size', type=int, required=False, default=256)
    return parser.parse_args()

def main():
  args = parse_args()
  client = TestClient('dali_ensemble', ['INPUT_0', 'INPUT_1'], ['OUTPUT_0', 'OUTPUT_1'], args.url,
                      concurrency=args.concurrency)
  client.run_tests(random_gen(args.max_batch_size), ref_func,
                   n_infers=args.n_iters, eps=1e-4)

if __name__ == '__main__':
  main()
