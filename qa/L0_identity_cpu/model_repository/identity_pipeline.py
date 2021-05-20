# The MIT License (MIT)
#
# Copyright (c) 2021 NVIDIA CORPORATION
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

import nvidia.dali as dali


def _parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Serialize the pipeline and save it to a file")
    parser.add_argument('file_path', type=str, help='The path where to save the serialized pipeline')
    return parser.parse_args()


@dali.pipeline_def(batch_size=3, num_threads=1, device_id=None)
def pipe():
    data = dali.fn.external_source(device="cpu", name="DALI_INPUT_0")
    return data


def main(filename):
    pipe().serialize(filename=filename)


if __name__ == '__main__':
    args = _parse_args()
    main(args.file_path)
