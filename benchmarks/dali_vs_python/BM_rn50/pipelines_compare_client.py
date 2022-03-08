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

import argparse, os, sys
import numpy as np
import tritonclient.grpc
import inspect
import re
from tqdm import tqdm
from PIL import Image

FLAGS = None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', action="store_true", required=False, default=False,
                        help='Enable verbose output')
    parser.add_argument('-u', '--url', type=str, required=False, default='localhost:8001',
                        help='Inference server URL. Default is localhost:8001.')
    parser.add_argument('-b', '--batch_size', type=int, required=False, default=1,
                        help='Batch size')
    parser.add_argument('--n_iter', type=int, required=False, default=-1,
                        help='Number of iterations , with `batch_size` size')
    parser.add_argument('-m', '--model_name', type=str, required=False, default="mydali",
                        help='Model name')
    parser.add_argument('-i', '--input_name', type=str, required=False, default="INPUT",
                        help='Input name')
    parser.add_argument('-o', '--output_name', type=str, required=False, default="OUTPUT",
                        help='Output name')
    parser.add_argument('--statistics', action='store_true', required=False, default=False,
                        help='Print tritonserver statistics after inferring')
    img_group = parser.add_mutually_exclusive_group()
    img_group.add_argument('--img', type=str, required=False, default=None,
                           help='Run a img dali pipeline. Arg: path to the image.')
    img_group.add_argument('--img_dir', type=str, required=False, default=None,
                           help='Directory, with images that will be broken down into batches and '
                                'inferred. The directory must contain images only')
    return parser.parse_args()


def _select_batch_size(batch_size_provider, batch_idx):
    if inspect.isgenerator(batch_size_provider):
        return next(batch_size_provider)
    elif isinstance(batch_size_provider, list):
        return batch_size_provider[batch_idx % len(batch_size_provider)]
    elif isinstance(batch_size_provider, int):
        return batch_size_provider
    raise TypeError("Incorrect batch_size_provider type. Actual: ", type(batch_size_provider))


def batcher(dataset, batch_size_provider, n_iterations=-1):
    """
    Generator, that splits dataset into batches with given batch size

    :param dataset: ndarray, where the outermost dimension is the size of the dataset. I.e.
                    ``dataset[0]`` is the first sample, ``dataset[1]`` is the second etc.
    :param batch_size_provider: Provides sizes for every batch.
                                * If the argument is a scalar, every batch will have the same size.
                                * If a list, sizes will be picked consecutively
                                (round-robin if necessary).
                                * If a generator, every batch size will be determined by a
                                ``next(batch_size_provider)`` call.
    :param n_iterations: Requested number of iterations. Samples gathered in ``dataset`` argument
                         will be used in round-robin manner to form the proper number of batches.
                         Default value (-1) means, that number of iterations will be inferred from
                         the dataset size - every sample will be used at most one time and last
                         (incomplete) batch will be dropped.
    :return: Yields batches
    """
    iter_idx = 0
    curr_sample = 0
    while True:
        try:
            batch_size = _select_batch_size(batch_size_provider, iter_idx)
        except StopIteration:
            return
        dataset_size = dataset.shape[0]

        # Stop condition
        if n_iterations == -1:
            if curr_sample + batch_size >= dataset_size:
                return
        else:
            if iter_idx >= n_iterations:
                return

        if curr_sample + batch_size < dataset_size:
            yield dataset[curr_sample: curr_sample + batch_size]
        else:
            # Get as many samples from this revolution of the dataset as possible,
            # then repeat the dataset as many revolutions as needed
            # and finally take the remaining samples from the last revolution
            suffix = dataset_size - curr_sample
            n_rep = (batch_size - suffix) // dataset_size
            prefix = batch_size - (suffix + dataset_size * n_rep)
            yield np.concatenate(
                (dataset[curr_sample:],
                 np.repeat(dataset, repeats=n_rep, axis=0),
                 dataset[:prefix])
            )
        curr_sample = (curr_sample + batch_size) % dataset_size
        iter_idx += 1


def load_image(img_path: str):
    """
    Loads image as an encoded array of bytes.
    This is a typical approach you want to use in DALI backend
    """
    with open(img_path, "rb") as f:
        img = f.read()
        return np.array(list(img)).astype(np.uint8)


def load_images(dir_path: str, name_pattern='.', max_images=-1):
    """
    Loads all files in given dir_path. Treats them as images. Optionally apply regex pattern to
    file names and use only the files, that suffice the pattern
    """
    assert max_images > 0 or max_images == -1
    images = []

    # Traverses directory for files (not dirs) and returns full paths to them
    path_generator = (os.path.join(dir_path, f) for f in os.listdir(dir_path) if
                      os.path.isfile(os.path.join(dir_path, f)) and
                      re.search(name_pattern, f) is not None)

    img_paths = [dir_path] if os.path.isfile(dir_path) else list(path_generator)
    if 0 < max_images < len(img_paths):
        img_paths = img_paths[:max_images]
    for img in tqdm(img_paths, desc="Reading images"):
        images.append(load_image(img))
    return images


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


def save_byte_image(bytes, size_wh=(224, 224), name_suffix=0):
    """
    Utility function, that can be used to save byte array as an image
    """
    im = Image.frombytes("RGB", size_wh, bytes, "raw")
    im.save("result_img_" + str(name_suffix) + ".jpg")


def generate_inputs(input_name, input_shape, input_dtype):
    return [tritonclient.grpc.InferInput(input_name, input_shape, input_dtype)]


def generate_outputs(output_name):
    return [tritonclient.grpc.InferRequestedOutput(output_name)]


def infer_dali(triton_client, batch):
    inputs = generate_inputs("DALI_INPUT_0", batch.shape, "UINT8")
    outputs = generate_outputs("DALI_OUTPUT_0")

    # Initialize the data
    inputs[0].set_data_from_numpy(batch)

    # Test with outputs
    results = triton_client.infer(model_name="rn50_dali", inputs=inputs, outputs=outputs)

    # Get the output arrays from the results
    output0_data = results.as_numpy("DALI_OUTPUT_0")
    return output0_data


def infer_python(triton_client, batch):
    inputs = generate_inputs("PYTHON_INPUT_0", batch.shape, "UINT8")
    outputs = generate_outputs("PYTHON_OUTPUT_0")

    # Initialize the data
    inputs[0].set_data_from_numpy(batch)

    # Test with outputs
    results = triton_client.infer(model_name="rn50_python", inputs=inputs, outputs=outputs)

    # Get the output arrays from the results
    output0_data = results.as_numpy("PYTHON_OUTPUT_0")
    return output0_data


def main():
    FLAGS = parse_args()
    try:
        triton_client = tritonclient.grpc.InferenceServerClient(url=FLAGS.url,
                                                                verbose=FLAGS.verbose)
    except Exception as e:
        print("channel creation failed: " + str(e))
        sys.exit()

    print("Loading images")

    if FLAGS.img_dir:
        image_data = load_images(FLAGS.img_dir, max_images=FLAGS.batch_size * FLAGS.n_iter)
    elif FLAGS.img:
        image_data = [load_image(FLAGS.img)]
    else:
        print("No image specified")
        sys.exit(1)

    image_data = array_from_list(image_data)
    print("Images loaded")

    for batch in tqdm(batcher(image_data, FLAGS.batch_size, n_iterations=FLAGS.n_iter),
                      desc="Inferring", total=FLAGS.n_iter):
        output0_dali = infer_dali(triton_client, batch)
        output0_python = infer_python(triton_client, batch)
        np.save("dali_output", output0_dali)
        np.save("python_output", output0_python)

    print('PASS: infer')


if __name__ == '__main__':
    main()
