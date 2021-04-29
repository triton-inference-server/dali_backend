import nvidia.dali as dali
import multiprocessing as mp
import tritonclient.grpc as t_client
import numpy as np
from typing import Sequence
from itertools import cycle, islice


def type_to_string(dtype):
  if dtype == np.half:
    return "FP16"
  if dtype == np.single:
    return "FP32"
  elif dtype == np.double:
    return "FP64"

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

  def run_tests(self, data, compare_to, n_infers=-1):
    generator = data if n_infers < 1 else islice(cycle(data), n_infers)
    for batches in generator:
      results = self.run_inference(batches)
      ref = compare_to(*batches)
      assert(len(results) == len(ref))
      for out, ref_out in zip(results, ref):
        assert out.shape == ref_out.shape
        assert np.allclose(out, ref_out)
    

def ref_func(inp1, inp2):
  return inp1 * 2 / 3, inp2 * 3 / 2

def main():
  client = TestClient('dali_ensemble', ['INPUT0', 'INPUT1'], ['OUTPUT0', 'OUTPUT1'])
  batch1 = np.ones((4, 5), dtype=np.single)
  batch2 = np.ones((4, 3), dtype=np.single)
  outputs = client.run_inference((batch1, batch2))
  client.run_tests([(batch1, batch2)], ref_func, n_infers=3)
  print(outputs)

if __name__ == '__main__':
  main()