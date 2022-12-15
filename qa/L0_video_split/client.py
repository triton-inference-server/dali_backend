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
import argparse
import nvidia.dali.fn as fn
import nvidia.dali as dali
import multiprocessing as mp
import nvidia.dali.experimental.eager as eager
from glob import glob
from os import environ
from itertools import cycle

def get_dali_extra_path():
  return environ['DALI_EXTRA_PATH']

def input_gen(batch_size):
  filenames = glob(f'{get_dali_extra_path()}/db/video/[cv]fr/*.mp4')
  filenames = filter(lambda filename: 'mpeg4' not in filename, filenames)
  filenames = filter(lambda filename: 'hevc' not in filename, filenames)
  # print(list(filenames))
  filenames = cycle(filenames)
  while True:
    batch = []
    for _ in range(batch_size):
      batch.append(np.fromfile(next(filenames), dtype=np.uint8))
    yield [eager.pad(batch).as_array()]


FRAMES_PER_SEQUENCE = 5
OUT_WIDTH = 300
OUT_HEIGHT = 300

@dali.pipeline_def(num_threads=min(mp.cpu_count(), 4), device_id=0,
                   output_dtype=dali.types.UINT8, output_ndim=[5, 4, 1],
                   prefetch_queue_depth=1)
def pipeline():
  vid = fn.external_source(device='cpu', name='INPUT', ndim=1, dtype=dali.types.UINT8)
  seq = fn.experimental.decoders.video(vid, device='mixed')
  seq = fn.resize(seq, resize_x=OUT_WIDTH, resize_y=OUT_HEIGHT)
  original_sequence = seq
  seq = fn.pad(seq, axis_names='F', align=FRAMES_PER_SEQUENCE)

  return fn.reshape(seq, shape=[-1, FRAMES_PER_SEQUENCE, OUT_HEIGHT, OUT_WIDTH, 3], name='OUTPUT'), \
         original_sequence,                                                                         \
         vid


def _split_outer_dim(output):
    arrays = [output.at(i) for i in range(len(output.shape()))]
    return np.concatenate(arrays)

class RefFunc:
  def __init__(self, max_batch_size):
    self._pipeline = pipeline(batch_size=max_batch_size)
    self._pipeline.build()


  def __call__(self, vids):
    self._pipeline.feed_input("INPUT", vids)
    out1, out2, out3 = self._pipeline.run()
    return _split_outer_dim(out1.as_cpu()), _split_outer_dim(out2.as_cpu()), out3.as_array()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-u', '--url', type=str, required=False, default='localhost:8001',
                        help='Inference server GRPC URL. Default is localhost:8001.')
    parser.add_argument('-n', '--n_iters', type=int, required=False, default=1, help='Number of iterations')
    parser.add_argument('-c', '--concurrency', type=int, required=False, default=1,
                        help='Request concurrency level')
    parser.add_argument('-b', '--max_batch_size', type=int, required=False, default=2)
    return parser.parse_args()

def main():
  args = parse_args()
  # client = TestClient('model.dali', ['INPUT'], ['OUTPUT', 'OUTPUT_images', 'INPUT'], args.url,
  #                     concurrency=args.concurrency)
  # client.run_tests(input_gen(args.max_batch_size), RefFunc(args.max_batch_size),
  #                  n_infers=args.n_iters, eps=1e-4)

  ref_func = RefFunc(args.max_batch_size)
  for i, vids in zip(range(10), input_gen(args.max_batch_size)):
    ref_func(*vids)
    print("Pass iteration: ", i)

if __name__ == '__main__':
  main()
