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
import argparse
from pathlib import Path
from shutil import copyfile
import shutil
import base64 as b64
import json


dali_extra_path = os.getenv('DALI_EXTRA_PATH', None)
assert dali_extra_path is not None, "Please set DALI_EXTRA_PATH env variable."

images_dir = Path(dali_extra_path) / 'db' / 'single' / 'jpeg'
images_paths = list(images_dir.glob('**/*.jpg'))

sized_images = sorted([(os.stat(p).st_size, p) for p in images_paths])

# choose 16 smallest samples
chosen_set = [p for (_, p) in sized_images[:16]]

# choose medium sized image 
chosen_sample = sized_images[8][1]

def save_sample_input(sample, dir_name, input_name):
  Path(dir_name).mkdir(exist_ok=True)
  shutil.copy(sample, Path(dir_name) / Path(input_name))

def get_content(fpath):
  with fpath.open("rb") as f:
    content = f.read()
    return {
      'content' : {
        'b64': b64.b64encode(content).decode('utf-8')
      },
      'shape': [len(content)]
    }

def save_json_dataset(files, dataset_filename, input_name):
  contents = [get_content(fpath) for fpath in files]
  inputs = [{input_name: content} for content in contents]
  result_dict = {'data': inputs}
  with open(dataset_filename, 'w') as dataset_file:
    json.dump(result_dict, dataset_file)

def get_args():
  parser = argparse.ArgumentParser(description='Prepare perf_analyzer input data.')
  parser.add_argument('-d', '--directory-name', required=False, action='store', default='inputs-data',
                      help='Directory name to store a single sample data.')
  parser.add_argument('-i', '--input-name', required=False, action='store', default='input',
                      help='Input name.')
  parser.add_argument('-f', '--dataset-filename', required=False, action='store', default='dataset.json',
                      help='Name of the created JSON dataset.')
  return parser.parse_args()

def main(args):
  save_sample_input(chosen_sample, args.directory_name, args.input_name)
  save_json_dataset(chosen_set, args.dataset_filename, args.input_name)

if __name__ == '__main__':
  args = get_args()
  main(args)
