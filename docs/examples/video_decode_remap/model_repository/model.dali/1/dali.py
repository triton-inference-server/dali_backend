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

import multiprocessing as mp

import nvidia.dali as dali
import nvidia.dali.fn as fn
from nvidia.dali.plugin.triton import autoserialize

OUT_WIDTH = 1920
OUT_HEIGHT = 1080


@autoserialize
@dali.pipeline_def(batch_size=256, num_threads=min(mp.cpu_count(), 4), device_id=0,
                   output_dtype=dali.types.UINT8, output_ndim=[4])
def pipeline():
    """
    DALI Pipeline, that performs the following processing:
    1. INPUT - encoded video file.
    2. Decode the video.
    3. Introduce a distortion to showcase how the distortion can be applied or removed.
    4. OUTPUT - distorted and decoded video.
    """
    # Decode video
    data = fn.external_source(name="INPUT", dtype=dali.types.UINT8, ndim=1)
    vid = fn.experimental.decoders.video(data, device='mixed')

    # Resize to match sizes of Remap parameters. This step is artificial in real life case
    # you most probably do not want to resize the image before removing the distortion.
    vid = fn.resize(vid, resize_x=OUT_WIDTH, resize_y=OUT_HEIGHT)

    # Remove distortion.
    mapx = fn.external_source(name="MAPX", ndim=2, dtype=dali.types.FLOAT).gpu()
    mapy = fn.external_source(name="MAPY", ndim=2, dtype=dali.types.FLOAT).gpu()
    # Provided camera maps assume, that the (0,0) point is in the center of the image.
    # Therefore, we have to modify them to have the origin in the top-left corner.
    mapx = mapx - OUT_WIDTH * 0.5
    mapy = mapy - OUT_HEIGHT * 0.5
    vid = fn.experimental.remap(vid, mapx, mapy, pixel_origin='center')

    # Resize, so that the output is smaller.
    vid = fn.resize(vid, resize_x=320, resize_y=240, name="OUTPUT")
    return vid
