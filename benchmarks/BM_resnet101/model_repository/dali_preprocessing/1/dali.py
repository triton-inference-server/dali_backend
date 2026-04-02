# The MIT License (MIT)
#
# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES
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

from nvidia.dali import fn
from nvidia.dali import pipeline_def
from nvidia.dali.plugin.triton import autoserialize
import nvidia.dali.types as types


@autoserialize
@pipeline_def(batch_size=16, num_threads=4, device_id=0)
def dali_preprocessing_pipe():
    """
    DALI pre-processing pipeline definition.
    """
    encoded = fn.external_source(name="encoded")
    decoded = fn.experimental.decoders.video(encoded, device="mixed", name="original")
    preprocessed = fn.resize(decoded, resize_x=224, resize_y=224)
    preprocessed = fn.crop_mirror_normalize(
        preprocessed,
        dtype=types.FLOAT,
        output_layout="FCHW",
        crop=(224, 224),
        mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
        std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
        name="preprocessed",
    )
    return decoded, preprocessed  # split_along_outer_axis enabled
