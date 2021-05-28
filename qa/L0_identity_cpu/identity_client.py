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
from numpy.random import randint
import tritongrpcclient
from PIL import Image
import math

np.random.seed(100019)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', action="store_true", required=False, default=False,
                        help='Enable verbose output')
    parser.add_argument('-u', '--url', type=str, required=False, default='localhost:8001',
                        help='Inference server URL. Default is localhost:8001.')
    parser.add_argument('--batch_size', type=int, required=False, default=4,
                        help='Batch size')
    parser.add_argument('--n_iter', type=int, required=False, default=-1,
                        help='Number of iterations , with `batch_size` size')
    parser.add_argument('--model_name', type=str, required=False, default="dali_identity_cpu",
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


def batcher(dataset, max_batch_size, n_iterations=-1):
    """
    Generator, that splits dataset into batches with given batch size
    """
    iter_idx = 0
    data_idx = 0
    while data_idx < len(dataset):
        if 0 < n_iterations <= iter_idx:
            raise StopIteration
        batch_size = min(randint(1, max_batch_size), len(dataset) - data_idx)
        iter_idx += 1
        yield dataset[data_idx : data_idx + batch_size]
        data_idx += batch_size


def main():
    FLAGS = parse_args()
    try:
        triton_client = tritongrpcclient.InferenceServerClient(url=FLAGS.url, verbose=FLAGS.verbose)
    except Exception as e:
        print("channel creation failed: " + str(e))
        sys.exit(1)

    if not (triton_client.is_server_live() or
            triton_client.is_server_ready() or
            triton_client.is_model_ready(model_name=FLAGS.model_name)):
        print("Error connecting to server: Server live {}. Server ready {}. Model ready {}".format(
            triton_client.is_server_live, triton_client.is_server_ready,
            triton_client.is_model_ready(model_name=FLAGS.model_name)))
        sys.exit(1)

    model_name = FLAGS.model_name
    model_version = -1

    input_data = [randint(0, 255, size=randint(100), dtype='uint8') for _ in
                  range(randint(100) * FLAGS.batch_size)]
    input_data = array_from_list(input_data)

    # Infer
    outputs = []
    input_name = "DALI_INPUT_0"
    output_name = "DALI_OUTPUT_0"
    input_shape = list(input_data.shape)
    outputs.append(tritongrpcclient.InferRequestedOutput(output_name))

    for batch in batcher(input_data, FLAGS.batch_size):
        print("Input mean before backend processing:", np.mean(batch))
        input_shape[0] = np.shape(batch)[0]
        print("Batch size: ", input_shape[0])
        inputs = [tritongrpcclient.InferInput(input_name, input_shape, "UINT8")]
        # Initialize the data
        inputs[0].set_data_from_numpy(batch)

        # Test with outputs
        results = triton_client.infer(model_name=model_name,
                                      inputs=inputs,
                                      outputs=outputs)

        # Get the output arrays from the results
        output0_data = results.as_numpy(output_name)
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
