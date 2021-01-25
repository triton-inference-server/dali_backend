#!/usr/bin/env python

# The MIT License (MIT)
#
# Copyright (c) 2020 NVIDIA CORPORATION
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

import argparse, os, sys
import numpy as np
import tritonclient.grpc
from PIL import Image
import math


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', action="store_true", required=False, default=False,
                        help='Enable verbose output')
    parser.add_argument('-u', '--url', type=str, required=False, default='localhost:8001',
                        help='Inference server URL. Default is localhost:8001.')
    parser.add_argument('--batch_size', type=int, required=False, default=1,
                        help='Batch size')
    parser.add_argument('--n_iter', type=int, required=False, default=-1,
                        help='Number of iterations , with `batch_size` size')
    parser.add_argument('--model_name', type=str, required=False, default="dali_multi_input",
                        help='Model name')
    return parser.parse_args()


def array_from_list(arrays):
    """
    Convert list of ndarrays to single ndarray with ndims+=1
    """
    lengths = list(map(lambda x, arr=arrays: arr[x].shape[0], [x for x in range(len(arrays))]))
    max_len = max(lengths)
    arrays = list(map(lambda arr, ml=max_len: np.pad(arr, ((0, ml - arr.shape[0]))), arrays))
    for arr in arrays:
        assert arr.shape == arrays[0].shape, "Arrays must have the same shape"
    return np.stack(arrays)


def batcher(dataset, batch_size, n_iterations=-1):
    """
    Generator, that splits dataset into batches with given batch size
    """
    assert len(dataset) % batch_size == 0
    n_batches = len(dataset) // batch_size
    iter_idx = 0
    for i in range(n_batches):
        if 0 < n_iterations <= iter_idx:
            raise StopIteration
        iter_idx += 1
        yield dataset[i * batch_size:(i + 1) * batch_size]


def main():
    FLAGS = parse_args()
    try:
        triton_client = tritonclient.grpc.InferenceServerClient(url=FLAGS.url, verbose=FLAGS.verbose)
    except Exception as e:
        print("channel creation failed: " + str(e))
        sys.exit(1)

    model_name = FLAGS.model_name
    model_version = -1

    input_data = [np.random.randint(0, 255, size=np.random.randint(100), dtype='uint8') for _ in
                  range(np.random.randint(100) * FLAGS.batch_size)]
    input_data = array_from_list(input_data)

    # Infer
    inputs = []
    outputs = []
    input_names = ["DALI_X_INPUT", "DALI_Y_INPUT"]
    output_names = ["DALI_OUTPUT_X", "DALI_OUTPUT_Y"]
    input_shape = list(input_data.shape)
    input_shape[0] = FLAGS.batch_size
    for iname in input_names:
        inputs.append(tritonclient.grpc.InferInput(iname, input_shape, "UINT8"))
    for oname in output_names:
        outputs.append(tritonclient.grpc.InferRequestedOutput(oname))

    for batch in batcher(input_data, FLAGS.batch_size):
        print("Input mean before backend processing:", np.mean(batch))
        # Initialize the data
        inputs[0].set_data_from_numpy(np.copy(batch))
        inputs[1].set_data_from_numpy(np.copy(batch))

        # Test with outputs
        results = triton_client.infer(model_name=model_name,
                                      inputs=inputs,
                                      outputs=outputs)

        # Get the output arrays from the results
        for oname in output_names:
            output0_data = results.as_numpy(oname)
            print("Output mean after backend processing:", np.mean(output0_data))
            print("Output shape: ", np.shape(output0_data))
            if not math.isclose(np.mean(output0_data), np.mean(batch)):
                print("Pre/post average does not match")
                sys.exit(1)
            else:
                print("pass")

    statistics = triton_client.get_inference_statistics(model_name=model_name)
    if len(statistics.model_stats) != 1:
        print("FAILED: Inference Statistics")
        sys.exit(1)


if __name__ == '__main__':
    main()
