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

import tritonclient.grpc as t_client
import numpy as np
from typing import Sequence
from itertools import cycle, islice
from numpy.random import randint
import argparse

# TODO: Extend and move to a separate file
def type_to_string(dtype):
  if dtype == np.half:
    return "FP16"
  if dtype == np.single:
    return "FP32"
  elif dtype == np.double:
    return "FP64"

# TODO: Extend and move to a separate file
class TestClient:
  def __init__(self, model_name: str, input_names: Sequence[str], output_names: Sequence[str],
               url, concurrency=1, verbose=False):
    self.client = t_client.InferenceServerClient(url=url, verbose=verbose)
    self.input_names = input_names
    self.output_names = output_names
    self.concurrency = concurrency
    self.model_name = model_name

  @staticmethod
  def _get_input(batch, name):
    inp = t_client.InferInput(name, list(batch.shape), type_to_string(batch.dtype))
    inp.set_data_from_numpy(batch)
    return inp

  def run_inference(self, batches):
    assert(len(batches) == len(self.input_names))
    if (len(batches) > 1):
      for b in batches:
        assert b.shape[0] == batches[0].shape[0]
    inputs = [self._get_input(batch, name) for batch, name in zip(batches, self.input_names)]
    outputs = [t_client.InferRequestedOutput(name) for name in self.output_names]
    results = self.client.infer(model_name=self.model_name, inputs=inputs, outputs=outputs)
    return [results.as_numpy(name) for name in self.output_names]

  def run_tests(self, data, compare_to, n_infers=-1, eps=1e-7):
    generator = data if n_infers < 1 else islice(cycle(data), n_infers)
    for it, batches in enumerate(generator):
      results = self.run_inference(batches)
      ref = compare_to(*batches)
      assert(len(results) == len(ref))
      for out_i, (out, ref_out) in enumerate(zip(results, ref)):
        assert out.shape == ref_out.shape
        if not np.allclose(out, ref_out, atol=eps):
          print("Test failure in iteration", it)
          print("Output", out_i)
          print("Expected:\n", ref_out)
          print("Actual:\n", out)
          assert False
      print('PASS iteration:', it)


# TODO: Use actual DALI pipelines to calculate ground truth
def ref_func(inp1, inp2):
  return inp1 * 2 / 3, (inp2 * 3).astype(np.half).astype(np.single) / 2

def random_gen(max_batch_size):
  while True:
    bs = randint(1, max_batch_size + 1)
    size1 = randint(100, 300)
    size2 = randint(100, 300)
    yield np.random.random((bs, size1)).astype(np.single), \
          np.random.random((bs, size2)).astype(np.single)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-u', '--url', type=str, required=False, default='localhost:8001',
                        help='Inference server GRPC URL. Default is localhost:8001.')
    parser.add_argument('--n_iters', type=int, required=False, default=1, help='Number of iterations')
    parser.add_argument('-b', '--max_batch_size', type=int, required=False, default=256)
    return parser.parse_args()

def main():
  args = parse_args()
  client = TestClient('dali_ensemble', ['INPUT0', 'INPUT1'], ['OUTPUT0', 'OUTPUT1'], args.url)
  client.run_tests(random_gen(args.max_batch_size), ref_func, n_infers=args.n_iters, eps=1e-4)

if __name__ == '__main__':
  main()
