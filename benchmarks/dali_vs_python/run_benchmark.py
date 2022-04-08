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

import argparse
import datetime
import os
import pandas as pd
import subprocess
import tempfile
import tritonclient.grpc as tritonclient

timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_name', type=str, required=True, help='Model name.',
                        choices=['rn50', 'jasper'])
    parser.add_argument('-u', '--url', type=str, required=False, default='localhost:8001',
                        help='Inference server URL. Default is localhost:8001.')
    parser.add_argument('-s', '--scenario', type=str, required=True, help='Model name.',
                        choices=['online', 'offline'])
    return parser.parse_args()


def input_data_path(model_name):
    return os.path.join(f'BM_{model_name}', 'test_sample')


def create_cmd(cmd_options):
    cmd = ["perf_analyzer"]
    for option, value in cmd_options.items():
        cmd.extend([option, value])
    return cmd


def check_sample_shape(input_name, model_name):
    input_data_dir = input_data_path(model_name=model_name)
    ret = subprocess.run(['stat', '--printf', '%s', f'{input_data_dir}/{input_name}'],
                         capture_output=True, check=True)
    return int(ret.stdout)


def create_cmd_options(model_name, input_name, batch_size, sample_shape, backend,
                       concurrency_range):
    input_data_dir = input_data_path(model_name=model_name)
    cmd_options = {
        '-m': f'{model_name}_{backend}',
        '-b': batch_size,
        '--input-data': input_data_dir,
        '--shape': f'{input_name}:{sample_shape}',
        '--concurrency-range': concurrency_range,
    }
    return cmd_options


def merge_csvs(csv_fd_list, output_filename, batch_sizes):
    df_per_file = list(pd.read_csv(f.name, sep=',') for f in csv_fd_list)
    for df, bs in zip(df_per_file, batch_sizes):
        df.insert(0, 'Batch Size', bs)
    df_merged = pd.concat(df_per_file, ignore_index=True)
    df_merged.to_csv(output_filename)


def run_perf_analyzer(model_name: str, input_name: str, backend: str, batch_sizes: list,
                      concurrency_range: str):
    sample_shape = check_sample_shape(input_name=input_name, model_name=model_name)
    result_files = []
    for bs in batch_sizes:
        cmd_options = create_cmd_options(model_name=model_name, input_name=input_name,
                                         batch_size=str(bs), sample_shape=sample_shape,
                                         backend=backend, concurrency_range=concurrency_range)
        results_file = tempfile.NamedTemporaryFile(mode='w')
        result_files.append(results_file)
        cmd_options['-f'] = results_file.name
        cmd = create_cmd(cmd_options=cmd_options)
        subprocess.run(cmd, capture_output=True, encoding='utf-8', check=True)
    output_filename = f"result_{model_name}_{backend}_{timestamp}.csv"
    merge_csvs(csv_fd_list=result_files,
               output_filename=output_filename,
               batch_sizes=batch_sizes)
    return output_filename


def analyze_model(tritonclient, model_name, input_name, model_backend, batch_sizes, scenario,
                  concurrency_range):
    tritonclient.load_model(f'{model_name}_{model_backend}')
    results_filename = run_perf_analyzer(model_name, input_name, model_backend, batch_sizes,
                                         concurrency_range)
    tritonclient.unload_model(f'{model_name}_{model_backend}')
    print(f"{model_name}_{model_backend} model results: {results_filename}")


def run_benchmark(model_descrs, batch_sizes, scenario, concurrency_range):
    client = tritonclient.InferenceServerClient(url='localhost:8001')
    for descr in model_descrs:
        analyze_model(client, descr['model_name'], descr['input_name'], descr['backend'],
                      batch_sizes, scenario=scenario, concurrency_range=concurrency_range)


def main():
    args = parse_args()
    model_descriptors = [
        {
            'model_name': f'{args.model_name}',
            'input_name': 'DALI_INPUT_0',
            'backend': 'dali',
        },
        {
            'model_name': f'{args.model_name}',
            'input_name': 'PYTHON_INPUT_0',
            'backend': 'python',
        }
    ]
    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256] if args.scenario == 'offline' else [1]
    concurrency_range = '1' if args.scenario == 'offline' else '1:32:1'
    run_benchmark(model_descriptors, batch_sizes, args.scenario, concurrency_range)


if __name__ == '__main__':
    main()
