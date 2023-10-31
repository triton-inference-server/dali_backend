#!/usr/bin/env python

# The MIT License (MIT)
#
# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES
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

import argparse
import os
import numpy as np
import tritonclient.grpc
from functools import partial
import queue


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-u',
                        '--url',
                        type=str,
                        required=False,
                        default='localhost:8001',
                        help='Inference server URL. Default is localhost:8001.')
    parser.add_argument(
        '--video',
        type=str,
        required=False,
        default=None,
        help='Path to a directory, where the video data is located.')
    return parser.parse_args()


class UserData:

    def __init__(self):
        self._completed_requests = queue.Queue()


def callback(usr_data, result, error):
    if error:
        usr_data._completed_requests.put(error)
    else:
        usr_data._completed_requests.put(result)


user_data = UserData()


def load_videos(filenames):
    """
    Loads all files in given directory path. Treats them as binary files.
    """
    return [np.fromfile(filename, dtype=np.uint8) for filename in filenames]


def array_from_list(arrays):
    """
    Convert list of ndarrays to single ndarray with ndims+=1. Pad if necessary.
    """
    lengths = list(
        map(lambda x, arr=arrays: arr[x].shape[0],
            [x for x in range(len(arrays))]))
    max_len = max(lengths)
    arrays = list(
        map(lambda arr, ml=max_len: np.pad(arr, (0, ml - arr.shape[0])),
            arrays))
    for arr in arrays:
        assert arr.shape == arrays[0].shape, "Arrays must have the same shape"
    return np.stack(arrays)


def main():
    FLAGS = parse_args()

    input_batch_size = 1  # `fn.inputs.video` accepts only one sample at the input,
    max_batch_size = 3  # but the output from `fn.inputs.video` has always max_batch_size.

    # Load test data
    if FLAGS.video is None:
        dali_extra_path = os.environ['DALI_EXTRA_PATH']
        filenames = [
            os.path.join(dali_extra_path, "db", "video", "containers", "mkv",
                         "cfr.mkv"),
        ]
    else:
        filenames = [
            os.path.join(FLAGS.video, p) for p in os.listdir(FLAGS.video)
        ]
        filenames = filenames[:input_batch_size]

    with tritonclient.grpc.InferenceServerClient(
            url=FLAGS.url) as triton_client:

        model_name = "model.dali"
        model_version = -1

        # Start stream
        triton_client.start_stream(callback=partial(callback, user_data))

        # Config output
        outputs = []
        output_name = "OUTPUT"
        outputs.append(tritonclient.grpc.InferRequestedOutput(output_name))

        # Config input 0: video
        video_raw = load_videos(filenames)
        video_raw = array_from_list(video_raw)
        input_shape = list(video_raw.shape)
        assert input_batch_size == input_shape[
            0], f"{input_batch_size} == {input_shape[0]}"

        # Config inputs 1 & 2: undistort (remap) maps
        npz = np.load('remap.npz')
        remap_u = [npz['remap_x'] for _ in range(max_batch_size)]
        remap_v = [npz['remap_y'] for _ in range(max_batch_size)]
        remap_u = array_from_list(remap_u)
        remap_v = array_from_list(remap_v)
        map_shape = list(remap_u.shape)

        # Initialize tritonserver inputs
        inputs = [
            tritonclient.grpc.InferInput("INPUT", input_shape, "UINT8"),
            tritonclient.grpc.InferInput("MAPX", map_shape, "FP32"),
            tritonclient.grpc.InferInput("MAPY", map_shape, "FP32"),
        ]

        inputs[0].set_data_from_numpy(video_raw)
        inputs[1].set_data_from_numpy(remap_u)
        inputs[2].set_data_from_numpy(remap_v)

        request_id = "0"
        triton_client.async_stream_infer(model_name=model_name,
                                         inputs=inputs,
                                         request_id=request_id,
                                         outputs=outputs)

        data_item = user_data._completed_requests.get()
        output0_data = data_item.as_numpy("OUTPUT")
        print(output0_data.shape)
        data_item = user_data._completed_requests.get()
        output0_data = data_item.as_numpy("OUTPUT")
        print(output0_data.shape)


if __name__ == '__main__':
    main()
