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

import os
import tarfile
from pathlib import Path
from typing import Tuple, Dict, List

from PIL import Image
from tqdm import tqdm

import time
import io

DATASETS_DIR = os.environ.get("DATASETS_DIR", None)
IMAGENET_DIRNAME = "imagenet"
IMAGE_ARCHIVE_FILENAME = "ILSVRC2012_img_val.tar"
DEVKIT_ARCHIVE_FILENAME = "ILSVRC2012_devkit_t12.tar.gz"
LABELS_REL_PATH = "ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt"
META_REL_PATH = "ILSVRC2012_devkit_t12/data/meta.mat"

TARGET_SIZE = (224, 224)  # (width, height)
_RESIZE_MIN = 256  # resize preserving aspect ratio to where this is minimal size


def parse_meta_mat(metafile) -> Dict[int, str]:
    import scipy.io

    meta = scipy.io.loadmat(metafile, squeeze_me=True)["synsets"]
    nums_children = list(zip(*meta))[4]
    meta = [meta[idx] for idx, num_children in enumerate(nums_children) if num_children == 0]
    idcs, wnids = list(zip(*meta))[:2]
    idx_to_wnid = {idx: wnid for idx, wnid in zip(idcs, wnids)}
    return idx_to_wnid


def _process_image(image_file, target_size):
    image = Image.open(image_file)
    original_size = image.size

    # scale image to size where minimal size is _RESIZE_MIN
    scale_factor = max(_RESIZE_MIN / original_size[0], _RESIZE_MIN / original_size[1])
    resize_to = int(original_size[0] * scale_factor), int(original_size[1] * scale_factor)
    resized_image = image.resize(resize_to)

    # central crop of image to target_size
    left, upper = (resize_to[0] - target_size[0]) // 2, (resize_to[1] - target_size[1]) // 2
    cropped_image = resized_image.crop((left, upper, left + target_size[0], upper + target_size[1]))
    return cropped_image


def main():
    import argparse

    parser = argparse.ArgumentParser(description="short_description")
    parser.add_argument(
        "--dataset-dir",
        help="Path to dataset directory where imagenet archives are stored and processed files will be saved.",
        required=False,
        default=DATASETS_DIR,
    )
    parser.add_argument(
      '--save',
      help='Save processed images.',
      required=False, default=False
    )
    parser.add_argument(
        "--target-size",
        help="Size of target image. Format it as <width>,<height>.",
        required=False,
        default=",".join(map(str, TARGET_SIZE)),
    )
    parser.add_argument(
      '--perf-file',
      required=False,
      default=None,
      help='Path to save a file with time measurements.'
    )
    args = parser.parse_args()

    if args.dataset_dir is None:
        raise ValueError(
            "Please set $DATASETS_DIR env variable to point dataset dir with original dataset archives "
            "and where processed files should be stored. Alternatively provide --dataset-dir CLI argument"
        )

    datasets_dir = Path(args.dataset_dir)
    target_size = tuple(map(int, args.target_size.split(",")))

    image_archive_path = datasets_dir / IMAGE_ARCHIVE_FILENAME
    if not image_archive_path.exists():
        raise RuntimeError(
            f"There should be {IMAGE_ARCHIVE_FILENAME} file in {datasets_dir}."
            f"You need to download the dataset from http://www.image-net.org/download."
        )

    devkit_archive_path = datasets_dir / DEVKIT_ARCHIVE_FILENAME
    if not devkit_archive_path.exists():
        raise RuntimeError(
            f"There should be {DEVKIT_ARCHIVE_FILENAME} file in {datasets_dir}."
            f"You need to download the dataset from http://www.image-net.org/download."
        )

    with tarfile.open(devkit_archive_path, mode="r") as devkit_archive_file:
        labels_file = devkit_archive_file.extractfile(LABELS_REL_PATH)
        labels = list(map(int, labels_file.readlines()))

        # map validation labels (idxes from LABELS_REL_PATH) into WNID compatible with training set
        meta_file = devkit_archive_file.extractfile(META_REL_PATH)
        idx_to_wnid = parse_meta_mat(meta_file)
        labels_wnid = [idx_to_wnid[idx] for idx in labels]

        # remap WNID into index in sorted list of all WNIDs - this is how network outputs class
        available_wnids = sorted(set(labels_wnid))
        wnid_to_newidx = {wnid: new_cls for new_cls, wnid in enumerate(available_wnids)}
        labels = [wnid_to_newidx[wnid] for wnid in labels_wnid]
    if args.perf_file is None:
      perf = False
    else:
      times = []
      perf = True
    output_dir = datasets_dir / IMAGENET_DIRNAME
    with tarfile.open(image_archive_path, mode="r") as image_archive_file:
        image_rel_paths = sorted(image_archive_file.getnames())
        for cls, image_rel_path in tqdm(zip(labels, image_rel_paths), total=len(image_rel_paths)):
            output_path = output_dir / str(cls) / image_rel_path
            original_image_file = image_archive_file.extractfile(image_rel_path)
            file_data = original_image_file.read()
            start = time.perf_counter()
            processed_image = _process_image(io.BytesIO(file_data), target_size)
            end = time.perf_counter()
            if perf:
              times.append(end-start)
            if args.save:
              output_path.parent.mkdir(parents=True, exist_ok=True)
              processed_image.save(output_path.as_posix())

    if perf:
      with open(args.perf_file, 'w') as perf_file:
        print(times, file=perf_file)


if __name__ == "__main__":
    main()
