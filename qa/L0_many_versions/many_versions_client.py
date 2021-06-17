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
import tritongrpcclient
from PIL import Image
import math


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', action="store_true", required=False, default=False,
                        help='Enable verbose output')
    parser.add_argument('-u', '--url', type=str, required=False, default='localhost:8001',
                        help='Inference server URL. Default is localhost:8001.')
    return parser.parse_args()



def main():
    FLAGS = parse_args()
    try:
        triton_client = tritongrpcclient.InferenceServerClient(url=FLAGS.url, verbose=FLAGS.verbose)
    except Exception as e:
        print("channel creation failed: " + str(e))
        sys.exit(1)

    if not (triton_client.is_server_live() or triton_client.is_server_ready()):
        print("Error connecting to server: Server live {}. Server ready {}.".format(
            triton_client.is_server_live(), triton_client.is_server_ready()))
        sys.exit(1)

    models_loaded = {
        "dali_all": [1, 2, 3],
        "dali_latest": [2, 3],
        "dali_specific": [2, 3],
    }

    models_not_loaded = {
        "dali_latest": [1],
        "dali_specific": [1],
    }

    for name, versions in models_loaded.items():
        for ver in versions:
            if not triton_client.is_model_ready(name, str(ver)):
                print("FAILED: Model {} version {} not ready".format(name, ver))
                sys.exit(1)

    for name, versions in models_not_loaded.items():
        for ver in versions:
            if triton_client.is_model_ready(name, str(ver)):
                print("FAILED: Model {} version {} incorrectly loaded".format(name, ver))
                sys.exit(1)


if __name__ == '__main__':
    main()
