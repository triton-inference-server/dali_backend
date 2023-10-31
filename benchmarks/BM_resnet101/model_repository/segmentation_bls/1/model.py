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

import json

import numpy as np
import torch
import triton_python_backend_utils as pb_utils


def run_inference(model_name, inputs, output_names):
    request = pb_utils.InferenceRequest(
        model_name=model_name,
        requested_output_names=output_names,
        inputs=inputs)
    response = request.exec()

    if response.has_error():
        raise pb_utils.TritonModelException(
            response.error().message())

    return map(lambda oname: pb_utils.get_output_tensor_by_name(response, oname), output_names)


def extract_subtensor(tensor, start_idx, size):
    tensor_pt = torch.from_dlpack(tensor.to_dlpack())
    subtensor = tensor_pt[start_idx: start_idx + size]
    return pb_utils.Tensor.from_dlpack(tensor.name(), torch.utils.dlpack.to_dlpack(subtensor))


class TritonPythonModel:
    def initialize(self, args):
        self.model_config = json.loads(args['model_config'])

    def execute(self, requests):
        responses = []
        for request in requests:
            in_encoded = pb_utils.get_input_tensor_by_name(request, "encoded")

            original, preprocessed = run_inference("dali_preprocessing", [in_encoded], ["original", "preprocessed"])

            probabilities, = run_inference("resnet101", [preprocessed], ["probabilities"])

            batch_size = original.shape()[0]
            in_height = original.shape()[1]
            in_width = original.shape()[2]
            video_height = pb_utils.Tensor("video_height", np.full((batch_size, 1), in_height, dtype=np.float32))
            video_width = pb_utils.Tensor("video_width", np.full((batch_size, 1), in_width, dtype=np.float32))

            segmented, = run_inference("dali_postprocessing", [original, probabilities, video_width, video_height],
                                       ["segmented"])

            inference_response = pb_utils.InferenceResponse(output_tensors=[original, segmented])
            responses.append(inference_response)

        return responses
