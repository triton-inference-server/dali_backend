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

import json

import cvcuda
import numpy as np
import torch
import torch.cuda.nvtx as nvtx
import torchnvjpeg
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def __init__(self):
        self.std = None
        self.mean = None

    def initialize(self, args):
        self.model_config = model_config = json.loads(args['model_config'])
        output0_config = pb_utils.get_output_config_by_name(model_config, "PYTHON_OUTPUT_0")
        self.output_dtype = pb_utils.triton_string_to_numpy(output0_config['data_type'])

    def execute(self, requests):
        responses = []

        for request in requests:
            in0 = pb_utils.get_input_tensor_by_name(request, "PYTHON_INPUT_0")
            npin = in0.as_numpy()
            in0_t = [npin[i].tobytes() for i in range(npin.shape[0])]
            out0_t = self.preprocess(in0_t, len(in0_t))
            out0_t = torch.as_tensor(out0_t.cuda(), device="cuda")
            out0_tensor = pb_utils.Tensor.from_dlpack("PYTHON_OUTPUT_0",
                                                      torch.utils.dlpack.to_dlpack(out0_t))

        response = pb_utils.InferenceResponse(output_tensors=[out0_tensor])
        responses.append(response)
        return responses

    def finalize(self):
        pass

    def preprocess(self, data_batch, batch_size, imgh=224, imgw=224):
        # Decode in batch using torchnvjpeg decoder on the GPU.
        decoder = torchnvjpeg.Decoder(
            0,
            0,
            True,
            0,
            batch_size,
            8,  # this is max_cpu_threads parameter. Not used internally.
            1920 * 1080 * 3,  # Max image size
            torch.cuda.current_stream(0),
        )
        nvtx.range_push("CV-CUDA DECODE")
        image_tensor_list = decoder.batch_decode(data_batch)
        nvtx.range_pop()

        # Convert the list of tensors to a tensor itself.
        nvtx.range_push("CV-CUDA STACK")
        image_tensors = torch.stack(image_tensor_list)
        nvtx.range_pop()

        # A torch tensor can be wrapped into a CVCUDA Object using the "as_tensor"
        # function in the specified layout. The datatype and dimensions are derived
        # directly from the torch tensor.
        nvtx.range_push("CV-CUDA MAKE TENSOR")
        cvcuda_input_tensor = cvcuda.as_tensor(image_tensors, "NHWC")
        nvtx.range_pop()

        # Resize
        nvtx.range_push("CV-CUDA RESIZE")
        cvcuda_resize_tensor = cvcuda.resize(
            cvcuda_input_tensor,
            (
                batch_size,
                imgh,
                imgw,
                3,
            ),
            cvcuda.Interp.LINEAR,
        )
        nvtx.range_pop()

        # Convert to the data type and range of values needed by the input layer
        # i.e uint8->float. A Scale is applied to normalize the values in the
        # range 0-1
        nvtx.range_push("CV-CUDA CAST")
        cvcuda_convert_tensor = cvcuda.convertto(
            cvcuda_resize_tensor, np.float32, scale=1 / 255
        )
        nvtx.range_pop()

        """
        The input to the network needs to be normalized based on the mean and
        std deviation value to standardize the input data.
        """

        # Create a torch tensor to store the mean and standard deviation
        # values for R,G,B
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        base_tensor = torch.Tensor(mean)
        stddev_tensor = torch.Tensor(std)

        # Reshape the the number of channels. The R,G,B values scale and offset
        # will be applied to every color plane respectively across the batch
        nvtx.range_push("CV-CUDA RESHAPE")
        base_tensor = torch.reshape(base_tensor, (1, 1, 1, 3)).cuda()
        stddev_tensor = torch.reshape(stddev_tensor, (1, 1, 1, 3)).cuda()
        nvtx.range_pop()

        # Wrap the torch tensor in a CVCUDA Tensor
        cvcuda_base_tensor = cvcuda.as_tensor(base_tensor, "NHWC")
        cvcuda_scale_tensor = cvcuda.as_tensor(stddev_tensor, "NHWC")

        # Apply the normalize operator and indicate the scale values are
        # std deviation i.e scale = 1/stddev
        nvtx.range_push("CV-CUDA NORMALIZE")
        cvcuda_norm_tensor = cvcuda.normalize(
            cvcuda_convert_tensor,
            base=cvcuda_base_tensor,
            scale=cvcuda_scale_tensor,
            flags=cvcuda.NormalizeFlags.SCALE_IS_STDDEV,
        )
        nvtx.range_pop()

        # The final stage in the preprocess pipeline includes converting the RGB
        # buffer into a planar buffer
        nvtx.range_push("CV-CUDA REFORMAT")
        cvcuda_preprocessed_tensor = cvcuda.reformat(cvcuda_norm_tensor, "NCHW")
        nvtx.range_pop()

        return cvcuda_preprocessed_tensor
