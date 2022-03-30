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

import torch
import torchvision
import torchvision.transforms as transforms
import triton_python_backend_utils as pb_utils

img_transforms = transforms.Compose(
    [transforms.Resize((224, 224)), transforms.CenterCrop(224), transforms.ToTensor()])


class TritonPythonModel:
    def __init__(self):
        self.std = None
        self.mean = None

    def initialize(self, args):
        self.model_config = model_config = json.loads(args['model_config'])
        output0_config = pb_utils.get_output_config_by_name(model_config, "PYTHON_OUTPUT_0")
        self.output_dtype = pb_utils.triton_string_to_numpy(output0_config['data_type'])

        with torch.no_grad():
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            self.mean = mean.cuda()
            self.std = std.cuda()

    def execute(self, requests):
        responses = []

        for request in requests:
            in0 = pb_utils.get_input_tensor_by_name(request, "PYTHON_INPUT_0")
            in0_t = torch.Tensor(in0.as_numpy())
            out0 = []
            for inp in in0_t:
                out0.append(self.decode_resize(inp.to(torch.uint8)))
            out0_t = torch.stack(out0)
            out0_t = self.normalize(out0_t)
            out0_tensor = pb_utils.Tensor.from_dlpack("PYTHON_OUTPUT_0",
                                                      torch.utils.dlpack.to_dlpack(out0_t))

        response = pb_utils.InferenceResponse(output_tensors=[out0_tensor])
        responses.append(response)
        return responses

    def finalize(self):
        pass

    def decode_resize(self, encoded_image):
        img = torchvision.io.decode_image(encoded_image)
        return img_transforms(transforms.functional.to_pil_image(img))

    def normalize(self, batch):
        with torch.no_grad():
            batch = batch.cuda()
            batch = batch.float()
            processed_batch = batch.unsqueeze(0).sub_(self.mean).div_(self.std)[0]
        return processed_batch
