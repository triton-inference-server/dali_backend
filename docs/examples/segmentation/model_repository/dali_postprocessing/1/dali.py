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
@pipeline_def(batch_size=32, num_threads=4, device_id=0, output_ndim=[3], output_dtype=[types.UINT8])
def dali_postprocessing_pipe(class_idx=0, prob_threshold=0.6):
    """
    DALI post-processing pipeline definition
    Args:
        class_idx: Index of the class that shall be segmented. Shall be correlated with `seg_class_name` argument
                   in the Model instance.
        prob_threshold: Probability threshold, at which the class affiliation is determined.

    Returns:
        Segmented images.
    """
    image = fn.external_source(device="gpu", name="original")
    image = fn.reshape(image, layout="HWC")  # No reshape performed, only setting the layout
    width = fn.external_source(device="cpu", name="video_width")
    height = fn.external_source(device="cpu", name="video_height")
    prob = fn.external_source(device="gpu", name="probabilities")
    prob = fn.reshape(prob, layout="CHW")  # No reshape performed, only setting the layout
    prob = fn.expand_dims(prob[class_idx], axes=[2], new_axis_names="C")
    prob = fn.resize(prob, resize_x=width, resize_y=height, interp_type=types.DALIInterpType.INTERP_NN)
    mask = fn.cast(prob > prob_threshold, dtype=types.UINT8)
    return image * mask
