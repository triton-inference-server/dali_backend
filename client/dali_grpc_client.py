#!/usr/bin/env python

# The MIT License (MIT)
#
# Copyright (c) 2020-2021 NVIDIA CORPORATION
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
import utils
import re
from tqdm import tqdm
from PIL import Image


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
    parser.add_argument('-m', '--model_name', type=str, required=False, default="dali_backend",
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


def main():
    FLAGS = parse_args()
    try:
        triton_client = tritonclient.grpc.InferenceServerClient(url=FLAGS.url,
                                                                verbose=FLAGS.verbose)
    except Exception as e:
        print("channel creation failed: " + str(e))
        sys.exit()

    model_name = FLAGS.model_name
    model_version = -1

    print("Loading images")

    image_data = load_images(FLAGS.img_dir if FLAGS.img_dir is not None else FLAGS.img,
                             max_images=FLAGS.batch_size * FLAGS.n_iter)

    image_data = array_from_list(image_data)
    print("Images loaded")

    for batch in tqdm(utils.batcher(image_data, FLAGS.batch_size, n_iterations=FLAGS.n_iter),
                      desc="Inferring", total=FLAGS.n_iter):

        inputs = generate_inputs(FLAGS.input_name, batch.shape, "UINT8")
        outputs = generate_outputs(FLAGS.output_name)

        # Initialize the data
        inputs[0].set_data_from_numpy(batch)

        # Test with outputs
        results = triton_client.infer(model_name=model_name, inputs=inputs, outputs=outputs)

        # Get the output arrays from the results
        output0_data = results.as_numpy(FLAGS.output_name)
        maxs = np.argmax(output0_data, axis=1)
        if FLAGS.statistics:
            for i in range(len(maxs)):
                print("Sample ", i, " - label: ", maxs[i], " ~ ", output0_data[i, maxs[i]])

    statistics = triton_client.get_inference_statistics(model_name="dali")
    if len(statistics.model_stats) != 1:
        print("FAILED: Inference Statistics")
        sys.exit(1)
    if FLAGS.statistics:
        print(statistics)

    print('PASS: infer')


if __name__ == '__main__':
    main()
