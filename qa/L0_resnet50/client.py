# The MIT License (MIT)
#
# Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import argparse
from model_repository.dali.pipeline import preprocessing

batch_size = 64

@pipeline_def
def dataloader(files_map, preprocess_data):
  files, labels = fn.readers.file(file_list=files_map)
  if preprocess_data:
    return preprocessing(files), labels
  else:
    return files, labels


def count_files(files_map):
  with open(files_map) as f:
    return sum(1 for line in f)


def get_args():
  parser = argparse.ArgumentParser(description='Test Resnet50 inference accuracy')
  parser.add_argument('-u', '--url', required=False, action='store', default='localhost:8001', help='Server url.')
  parser.add_argument('-m', '--model', required=True, action='store', help='Model name.')
  parser.add_argument('-p', '--preprocessing', required=False, action='store', default='false', 
                      help='Set to true if client-side preprocessing is needed.')
  parser.add_argument('-f', '--file_list', required=True, action='store', help='Path to a dataset files list.')
  return parser.parse_args()


def main(args):
  print(count_files(args.file_list))
  iters = (count_files(args.file_list) + batch_size - 1) // batch_size
  


if __name__ == '__main__':
  args = get_args()
  main(args)
