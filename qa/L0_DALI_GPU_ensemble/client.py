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

import nvidia.dali as dali
import multiprocessing as mp
import tritonclient.grpc as t_client
import numpy as np
from typing import Sequence
from itertools import cycle, islice
from numpy.random import randint

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
               url = 'localhost:8001', concurrency=1, verbose=False):
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
    it = -1
    for batches in generator:
      it += 1
      results = self.run_inference(batches)
      ref = compare_to(*batches)
      assert(len(results) == len(ref))
      for out, ref_out in zip(results, ref):
        assert out.shape == ref_out.shape
        if not np.allclose(out, ref_out, atol=eps):
          print("Test failure in iteration", it)
          print("Expected output:\n", ref_out)
          print("Actual output:\n", out)
          return


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

def main():
  client = TestClient('dali_ensemble', ['INPUT0', 'INPUT1'], ['OUTPUT0', 'OUTPUT1'])
  client.run_tests(random_gen(256), ref_func, n_infers=200)

if __name__ == '__main__':
  main()
