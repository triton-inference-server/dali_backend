# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
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
import pathlib
import model_navigator as nav
import torch

LOGGER = logging.getLogger("depoly_on_triton")


def get_model():
    efficientnet = torch.hub.load('/DeepLearningExamples', 'nvidia_efficientnet_b0', pretrained=True, source='local')
    efficientnet.eval().cuda()
    return efficientnet


def _get_args():
    parser = argparse.ArgumentParser(
        description="Script for exporting models from supported frameworks.",
        allow_abbrev=False,
    )
    parser.add_argument("--model-name", help="Name of model.", required=True)
    parser.add_argument(
        "--model-repository",
        help="Model repository for Triton deployment.",
        required=False,
    )
    parser.add_argument(
        "-v", "--verbose", help="Verbose logs.", action="store_true", default=False
    )
    parser.add_argument(
        "-b", "--batch-size", help="Maximum batch size for the inference scenario.", required=True, type=int,
    )
    args = parser.parse_args()

    return args


def main():
    args = _get_args()

    log_level = logging.INFO if not args.verbose else logging.DEBUG
    log_format = "%(asctime)s %(levelname)s %(name)s %(message)s"
    logging.basicConfig(level=log_level, format=log_format)

    LOGGER.info("args:")
    for key, value in vars(args).items():
        LOGGER.info(f"    {key} = {value}")

    model = get_model()

    profile = nav.OptimizationProfile()
    profile.batch_sizes = [args.batch_size]

    data_type = torch.float32
    precision = "fp32"

    package = nav.torch.optimize(
        model=model,
        dataloader=[torch.randn(1, 3, 224, 224, dtype=data_type) for _ in range(100)],
        input_names=("INPUT__0",),
        output_names=("OUTPUT__0",),
        custom_configs=[
            nav.OnnxConfig(
                dynamic_axes={"INPUT__0": {0: "batch"}},
            ),
            nav.TensorRTConfig(precision=(precision,)),
            nav.TorchTensorRTConfig(precision=(precision,)),
        ],
        verbose=True,
        optimization_profile=profile,
    )

    output_package = pathlib.Path.cwd() / "package.nav"
    nav.package.save(package=package, path=output_package, override=True)
    LOGGER.info(f"Package saved to {output_package}")

    if args.model_repository:
        nav.triton.model_repository.add_model_from_package(
            model_repository_path=pathlib.Path(args.model_repository),
            model_name=args.model_name,
            package=package,
            strategy=nav.MaxThroughputStrategy(),
        )
        LOGGER.info(f"Model deployment created in {args.model_repository}")


if __name__ == "__main__":
    main()
