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

import nvidia.dali as dali
import nvidia.dali.fn as fn
import multiprocessing as mp
from nvidia.dali.plugin.triton import autoserialize

FRAMES_PER_SEQUENCE = 5
OUT_WIDTH = 300
OUT_HEIGHT = 300

@autoserialize
@dali.pipeline_def(batch_size=256, num_threads=min(mp.cpu_count(), 4), device_id=0,
                   output_dtype=[dali.types.UINT8, dali.types.UINT8], output_ndim=[5, 1])
def pipeline():
  vid = fn.external_source(device='cpu', name='INPUT', ndim=1, dtype=dali.types.UINT8)
  seq = fn.experimental.decoders.video(vid, device='mixed')
  seq = fn.resize(seq, resize_x=OUT_WIDTH, resize_y=OUT_HEIGHT)
  seq = fn.pad(seq, axis_names='F', align=FRAMES_PER_SEQUENCE)

  return fn.reshape(seq, shape=[-1, FRAMES_PER_SEQUENCE, OUT_HEIGHT, OUT_WIDTH, 3], name='OUTPUT'), vid
