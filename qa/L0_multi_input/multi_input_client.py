#!/usr/bin/env python

# The MIT License (MIT)
#
# Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES
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
import tritonclient.grpc
from PIL import Image
import math

np.random.seed(100019)

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


def batcher(dataset, max_batch_size, n_iterations=-1):
    """
    Generator, that splits dataset into batches with given batch size
    """
    iter_idx = 0
    data_idx = 0
    while data_idx < len(dataset):
        if 0 < n_iterations <= iter_idx:
            raise StopIteration
        batch_size = min(randint(0, max_batch_size) + 1, len(dataset) - data_idx)
        iter_idx += 1
        yield dataset[data_idx : data_idx + batch_size]
        data_idx += batch_size


def main():
    FLAGS = parse_args()
    try:
        triton_client = tritonclient.grpc.InferenceServerClient(url=FLAGS.url, verbose=FLAGS.verbose)
    except Exception as e:
        print("channel creation failed: " + str(e))
        sys.exit(1)

    model_name = FLAGS.model_name
    model_version = -1

    input_data = [randint(0, 255, size=randint(100), dtype='uint8') for _ in
                  range(randint(100) * FLAGS.batch_size)]
    input_data = array_from_list(input_data)

    # Infer
    outputs = []
    input_names = ["DALI_X_INPUT", "DALI_Y_INPUT"]
    scalars_name = "DALI_SCALAR"
    output_names = ["DALI_unchanged", "DALI_changed"]

    input_shape = list(input_data.shape)

    for oname in output_names:
        outputs.append(tritonclient.grpc.InferRequestedOutput(oname))

    for batch in batcher(input_data, FLAGS.batch_size):
        print("Input mean before backend processing:", np.mean(batch))
        batch_size = np.shape(batch)[0]
        print("Batch size: ", batch_size)

        # Initialize the data
        input_shape[0] = batch_size
        scalars = randint(0, 1024, size=(batch_size, 1), dtype=np.int32)
        inputs = [tritonclient.grpc.InferInput(iname, input_shape, "UINT8") for iname in
                  input_names]
        scalar_input = tritonclient.grpc.InferInput(scalars_name, [batch_size, 1], "INT32")
        for inp in inputs:
            inp.set_data_from_numpy(np.copy(batch))
        scalar_input.set_data_from_numpy(scalars)

        # Test with outputs
        results = triton_client.infer(model_name=model_name, inputs=[*inputs, scalar_input], outputs=outputs)

        # Get the output arrays from the results
        for oname in output_names:
            print("\nOutput: ", oname)
            output_data = results.as_numpy(oname)
            print("Output mean after backend processing:", np.mean(output_data))
            print("Output shape: ", np.shape(output_data))
            expected = np.multiply(batch, 1 if oname is "DALI_unchanged" else scalars,
                                   dtype=np.int32)
            if not np.allclose(output_data, expected):
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
