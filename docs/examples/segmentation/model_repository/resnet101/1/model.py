# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging

import json
# import nvtx  # pytype: disable=import-error
import torch  # pytype: disable=import-error
from torchvision.models import segmentation as segmentation_models  # pytype: disable=import-error
import triton_python_backend_utils as pb_utils
from torch.utils.dlpack import from_dlpack


class SegmentationPyTorch:
    """
    Excerpt from CV-CUDA segmentation example:
    https://github.com/CVCUDA/CV-CUDA/blob/release_v0.3.x/samples/segmentation/python/model_inference.py
    """
    def __init__(self, seg_class_name, device_id):
        self.logger = logging.getLogger(__name__)
        self.device_id = device_id
        # Fetch the segmentation index to class name information from the weights
        # meta properties.
        # The underlying pytorch model that we use for inference is the FCN model
        # from torchvision.
        torch_model = segmentation_models.fcn_resnet101
        weights = segmentation_models.FCN_ResNet101_Weights.DEFAULT

        try:
            self.class_index = weights.meta["categories"].index(seg_class_name)
        except ValueError:
            raise ValueError(
                "Requested segmentation class '%s' is not supported by the "
                "fcn_resnet101 model. All supported class names are: %s"
                % (seg_class_name, ", ".join(weights.meta["categories"]))
            )

        # Inference uses PyTorch to run a segmentation model on the pre-processed
        # input and outputs the segmentation masks.
        class FCN_Softmax(torch.nn.Module):  # noqa: N801
            def __init__(self, fcn):
                super().__init__()
                self.fcn = fcn

            def forward(self, x):
                infer_output = self.fcn(x)["out"]
                return torch.nn.functional.softmax(infer_output, dim=1)

        fcn_base = torch_model(weights=weights)
        fcn_base.eval()
        self.model = FCN_Softmax(fcn_base).cuda(self.device_id)
        self.model.eval()

        self.logger.info("Using PyTorch as the inference engine.")

    def __call__(self, tensor):
        # nvtx.push_range("inference.torch")

        with torch.no_grad():
            segmented = self.model(tensor)

        # nvtx.pop_range()
        return segmented


class TritonPythonModel:
    def __init__(self):
        self.segmentation_model=SegmentationPyTorch(
            seg_class_name="__background__",
            device_id=0,
        )

    def initialize(self, args):
        self.model_config = model_config = json.loads(args['model_config'])
        output0_config = pb_utils.get_output_config_by_name(model_config, "probabilities")
        self.output_dtype = pb_utils.triton_string_to_numpy(output0_config['data_type'])



    def execute(self, requests):
        responses = []

        for request in requests:
            in0 = pb_utils.get_input_tensor_by_name(request, "preprocessed")
            in0_t = from_dlpack(in0.to_dlpack()).cuda()
            out0_t = self.segmentation_model(in0_t)
            out0 = pb_utils.Tensor.from_dlpack("probabilities", out0_t)

        response = pb_utils.InferenceResponse(output_tensors=[out0])
        responses.append(response)
        return responses
