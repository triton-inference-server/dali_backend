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
import features
import soundfile as sf
import io
import numpy as np
import triton_python_backend_utils as pb_utils


def decode_audio(audio_bytes):
    return sf.read(io.BytesIO(audio_bytes))


class TritonPythonModel:
    def __init__(self):
        pass

    def initialize(self, args):
        self.model_config = model_config = json.loads(args['model_config'])
        output0_config = pb_utils.get_output_config_by_name(model_config, "PYTHON_OUTPUT_0")
        self.output_dtype = pb_utils.triton_string_to_numpy(output0_config['data_type'])
        self.feat_proc = features.FilterbankFeatures(
            spec_augment=None, cutout_augment=None, sample_rate=16000, window_size=0.02,
            window_stride=0.01, window="hann", normalize="per_feature", n_fft=512, preemph=0.97,
            n_filt=64, lowfreq=0, highfreq=None, log=True, dither=1e-5, pad_align=16,
            pad_to_max_duration=False, max_duration=float('inf'), frame_splicing=1)

    def execute(self, requests):
        responses = []

        for request in requests:
            in0 = pb_utils.get_input_tensor_by_name(request, "PYTHON_INPUT_0")
            in0_t = in0.as_numpy()
            decoded = []
            for inp in in0_t:
                aud_sr = decode_audio(inp.tobytes())
                decoded.append((aud_sr[0], aud_sr[0].shape[0]))
            max_len = 0
            for dec in decoded:
                max_len = max_len if max_len > dec[1] else dec[1]
            audio = []
            audio_lens = []
            for aud, length in decoded:
                audio.append(aud)
                np.pad(audio[-1], (0, max_len - audio[-1].shape[0]))
                audio_lens.append(length)
            audio_array = np.array(audio)
            len_array = np.array(audio_lens)
            dec_t = torch.Tensor(audio_array)
            len_t = torch.Tensor(len_array)
            dec_t = dec_t.cuda()
            len_t = len_t.cuda()
            out_audio, out_len = self.feat_proc(dec_t, len_t)
            out0_tensor = pb_utils.Tensor.from_dlpack("PYTHON_OUTPUT_0",
                                                      torch.utils.dlpack.to_dlpack(out_audio))

        response = pb_utils.InferenceResponse(output_tensors=[out0_tensor])
        responses.append(response)
        return responses

    def finalize(self):
        pass
