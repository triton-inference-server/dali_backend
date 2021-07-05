# The MIT License (MIT)
#
# Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import tritonclient.grpc as t_client
import numpy as np
from typing import Sequence
from itertools import cycle, islice
from numpy.random import randint
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

# TODO: Extend
def type_to_string(dtype):
  if dtype == np.half:
    return "FP16"
  if dtype == np.single:
    return "FP32"
  elif dtype == np.double:
    return "FP64"

def grouper(n, iterable):
    it = iter(iterable)
    while True:
        chunk = tuple(islice(it, n))
        if not chunk:
            return
        yield chunk


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

  def test_infer(self, data, it):
    assert(len(data) == len(self.input_names))
    if (len(data) > 1):
      for b in data:
        assert b.shape[0] == data[0].shape[0]
    inputs = [self._get_input(batch, name) for batch, name in zip(data, self.input_names)]
    outputs = [t_client.InferRequestedOutput(name) for name in self.output_names]
    res = self.client.infer(model_name=self.model_name, inputs=inputs, outputs=outputs)
    res_data = [res.as_numpy(name) for name in self.output_names]
    return it, data, res_data

  def run_tests(self, data, compare_to, n_infers=-1, eps=1e-7):
    generator = data if n_infers < 1 else islice(cycle(data), n_infers)
    for pack in grouper(self.concurrency, enumerate(generator)):
      with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
        results_f = [executor.submit(self.test_infer, data, it) for it, data in pack]
        for future in as_completed(results_f):
          it, data, results = future.result()
          ref = compare_to(*data)
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
