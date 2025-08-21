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
import numpy as np
from model_repository.dali.pipeline import preprocessing
from dali_backend.test_utils.client import ClassificationClient

@pipeline_def
def dataloader(files_map, preprocess_data):
  files, labels = fn.readers.file(file_list=files_map)
  if preprocess_data:
    return preprocessing(files, device='cpu'), labels
  else:
    return fn.pad(files, fill_value=0), labels


def count_files(files_map):
  with open(files_map) as f:
    return sum(1 for line in f)


def get_args():
  parser = argparse.ArgumentParser(description='Test Resnet50 inference accuracy')
  parser.add_argument('-u', '--url', required=False, action='store', default='localhost:8001', help='Server url.')
  parser.add_argument('-m', '--model', required=True, action='store', help='Model name.')
  parser.add_argument('-c', '--concurrency', required=False, type=int, action='store', default=1, help='Client concurrency.')
  parser.add_argument('-b', '--batch_size', required=False, type=int, action='store', default=1, help='Request batch size.')
  parser.add_argument('-i', '--input_names', action='store', help='Names of the model inputs, separated with comma.')
  parser.add_argument('-o', '--output_names', action='store', help='Names of the model outputs, separated with comma.')
  parser.add_argument('-p', '--preprocessing', required=False, action='store', default='false', 
                      help='Set to true if client-side preprocessing is needed.')
  parser.add_argument('-f', '--file_list', required=True, action='store', help='Path to a dataset files list.')
  return parser.parse_args()


def data_generator(files_map, preprocessed, batch_size):
  pipeline = dataloader(files_map, preprocessed, batch_size=batch_size, num_threads=4, device_id=0)
  pipeline.build()
  while True:
    data, labels = pipeline.run()
    yield (data.as_array(), labels.as_array())

def main(args):
  print(count_files(args.file_list))
  iters = (count_files(args.file_list) + args.batch_size - 1) // args.batch_size
  inames = args.input_names.split(',')
  onames = args.output_names.split(',')
  print(inames)
  print(onames)
  # return
  preprocess = args.preprocessing == 'True' or args.preprocessing == 'true'
  client = ClassificationClient(args.model, inames, onames, args.url, args.concurrency)
  correct, incorrect = client.test_accuracy(data_generator(args.file_list, preprocess, args.batch_size), n_infers=1)
  print(correct, incorrect)


if __name__ == '__main__':
  args = get_args()
  main(args)
