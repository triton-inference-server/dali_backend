#!/usr/bin/env python

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

import argparse
import os
import sys
import numpy as np
import tritonclient.grpc


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-u', '--url', type=str, required=False, default='localhost:8001',
                        help='Inference server URL. Default is localhost:8001.')
    parser.add_argument('--video', type=str, required=False, default=None,
                        help='Path to a directory, where the video data is located.')
    return parser.parse_args()


def load_videos(filenames):
    """
    Loads all files in given directory path. Treats them as binary files.
    """
    return [np.fromfile(filename, dtype=np.uint8) for filename in filenames]


def array_from_list(arrays):
    """
    Convert list of ndarrays to single ndarray with ndims+=1. Pad if necessary.
    """
    lengths = list(map(lambda x, arr=arrays: arr[x].shape[0], [x for x in range(len(arrays))]))
    max_len = max(lengths)
    arrays = list(map(lambda arr, ml=max_len: np.pad(arr, (0, ml - arr.shape[0])), arrays))
    for arr in arrays:
        assert arr.shape == arrays[0].shape, "Arrays must have the same shape"
    return np.stack(arrays)


def main():
    FLAGS = parse_args()

    # Load test data
    if FLAGS.video is None:
        dali_extra_path = os.environ['DALI_EXTRA_PATH']
        filenames = [
            os.path.join(dali_extra_path, "db", "video", "containers", "mkv", "cfr-h264.mkv"),
            os.path.join(dali_extra_path, "db", "video", "containers", "mkv", "cfr-h265.mkv"),
        ]
    else:
        filenames = os.listdir(FLAGS.video)

    batch_size = len(filenames)

    try:
        triton_client = tritonclient.grpc.InferenceServerClient(url=FLAGS.url)
    except Exception as e:
        print("channel creation failed: " + str(e))
        sys.exit(1)

    model_name = "model.dali"
    model_version = -1

    # Config output
    outputs = []
    output_name = "OUTPUT"
    outputs.append(tritonclient.grpc.InferRequestedOutput(output_name))

    # Config input 0: video
    video_raw = load_videos(filenames)
    video_raw = array_from_list(video_raw)
    input_shape = list(video_raw.shape)
    assert batch_size == input_shape[0]

    # Config inputs 1 & 2: undistort (remap) maps
    npz = np.load('./remap.npz')
    remap_u = [npz['remap_u'] for _ in range(batch_size)]
    remap_v = [npz['remap_v'] for _ in range(batch_size)]
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

    print("Sending video to the inference server, waiting for the result...")
    results = triton_client.infer(model_name=model_name,
                                  inputs=inputs,
                                  outputs=outputs)

    # Get the output and show the result
    output0_data = results.as_numpy(output_name)
    print(output0_data.shape)

    # Show the distorted video (first one from the batch)
    import cv2
    for frame in output0_data[0]:
        cv2.imshow("Frame", frame)
        cv2.waitKey(100)


if __name__ == '__main__':
    main()
