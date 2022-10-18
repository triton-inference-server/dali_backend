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

import numpy as np
from dali_backend.test_utils.client import TestClient
from numpy.random import randint, random
import argparse

def ref_func(imgs):
  output = [np.fliplr(imgs[i]) for i in range(imgs.shape[0])]
  return np.stack(output),

def random_gen(max_batch_size):
  while True:
    bs = randint(1, max_batch_size + 1)
    width = randint(100, 200)
    height = randint(100, 200)
    yield [(random((bs, height, width, 3)) * 255).astype(np.uint8)]


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
  client = TestClient('img_proc.dali', ['DALI_INPUT_0'], ['DALI_OUTPUT_0',], args.url,
                      concurrency=args.concurrency)
  client.run_tests(random_gen(args.max_batch_size), ref_func,
                   n_infers=args.n_iters, eps=1e-4)

if __name__ == '__main__':
  main()
