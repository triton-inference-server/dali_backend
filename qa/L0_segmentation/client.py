#!/usr/bin/env python3
# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import logging

import numpy as np  # pytype: disable=import-error

from pytriton.client import ModelClient

logger = logging.getLogger("examples.dali_resnet101_pytorch.client")

VIDEO_PATH = "test_video/sintel_trailer_short.mp4"


def load_video(video_path):
    return np.array(np.fromfile(video_path, dtype=np.uint8)).reshape(1, -1)


def infer_model(named_inputs, args):
    with ModelClient(f"grpc://{args.url}", "segmentation_bls", init_timeout_s=args.init_timeout_s) as client:
        result_data = client.infer_batch(**named_inputs)

        original = result_data["original"]
        segmented = result_data["segmented"]

        if args.dump_images:
            for i, (orig, segm) in enumerate(zip(original, segmented)):
                import cv2  # pytype: disable=import-error

                cv2.imwrite(f"test_video/orig{i}.jpg", orig)
                cv2.imwrite(f"test_video/segm{i}.jpg", segm)

        logger.info("Processing finished.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '-u', "--url",
        default="localhost",
        help=(
            "Url to Triton server (ex. grpc://localhost:8001)."
            "HTTP protocol with default port is used if parameter is not provided"
        ),
        required=False,
    )
    parser.add_argument(
        "--init-timeout-s",
        type=float,
        default=600.0,
        help="Server and model ready state timeout in seconds.",
        required=False,
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--dump-images",
        action="store_true",
        default=False,
        help="If True, the client will save processed images to disk. Requires cv2 module.",
        required=False,
    )
    parser.add_argument(
        "--video-path",
        default=None,
        help="Paths of the video to process.",
        required=False,
    )
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s - %(levelname)s - %(name)s: %(message)s")

    named_inputs = {
        "encoded": load_video(VIDEO_PATH if args.video_path is None else args.video_path),
    }

    infer_model(named_inputs, args)


if __name__ == "__main__":
    main()
