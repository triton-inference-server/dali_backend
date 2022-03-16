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

import tritonclient.grpc as tritonclient
import argparse
import sys
import re
import subprocess


def create_cmd(cmd_options):
    cmd = ["perf_analyzer"]
    for option, value in cmd_options.items():
        cmd.extend([option, value])
    return cmd


def check_sample_shape(input_name):
    input_data_dir = 'test_sample'
    ret = subprocess.run(['stat', '--printf', '%s', f'{input_data_dir}/{input_name}'],
                         capture_output=True, check=True)
    return int(ret.stdout)


def create_cmd_options(model_name, input_name, batch_size, sample_shape):
    input_data_dir = 'test_sample'
    cmd_options = {
        '-m': model_name,
        '-b': batch_size,
        '--input-data': input_data_dir,
        '--shape': f'{input_name}:{sample_shape}',
    }
    return cmd_options


def parse_perf_analyzer_result(pa_result):
    def search(regex):
        return re.search(pattern=regex, string=pa_result).group()

    ret = {
        'batch_size': search(r'(?<=Batch size: )\d+'),
        'concurrency': search(r'(?<=Concurrency: )\d+'),
        'overhead': search(r'(?<=overhead )\d+'),
        'queue': search(r'(?<=queue )\d+'),
        'compute_input': search(r'(?<=compute input )\d+'),
        'compute_infer': search(r'(?<=compute infer )\d+'),
        'compute_output': search(r'(?<=compute output )\d+'),
        'server_latency': search(r'(?<=Avg request latency: )\d+'),
        'client_latency': search(r'(?<=Avg latency: )\d+'),
        'client_throughput': search(r'(?<=Throughput: )\d+')
    }
    return ret


def generate_csv(result_dicts):
    ret = ''
    for key in result_dicts[0].keys():
        ret += key
        ret += ','
    ret = ret[:-1]
    ret += '\n'
    for input_dict in result_dicts:
        for key in input_dict.values():
            ret += key
            ret += ','
        ret = ret[:-1]
        ret += '\n'
    return ret


def run_perf_analyzer(model_name: str, input_name: str, batch_sizes: list):
    sample_shape = check_sample_shape(input_name)
    ret_dicts = []
    for bs in batch_sizes:
        cmd_options = create_cmd_options(model_name, input_name, str(bs), sample_shape)
        cmd = create_cmd(cmd_options=cmd_options)
        pa_output = subprocess.run(cmd, capture_output=True, encoding='utf-8', check=True)
        ret_dicts.append(parse_perf_analyzer_result(pa_output.stdout))
    return generate_csv(result_dicts=ret_dicts)


def analyze_model(tritonclient, model_name, input_name, batch_sizes):
    tritonclient.load_model(model_name)
    stats = run_perf_analyzer(model_name, input_name, batch_sizes)
    tritonclient.unload_model(model_name)
    print(f"{model_name} model results:")
    print(stats)


def run_benchmark(model_descrs, batch_sizes):
    client = tritonclient.InferenceServerClient(url='localhost:8001')
    for descr in model_descrs:
        analyze_model(client, descr['model_name'], descr['input_name'], batch_sizes)


if __name__ == '__main__':
    model_descriptors = [
        {
            'model_name': 'rn50_dali',
            'input_name': 'DALI_INPUT_0',
        },
        {
            'model_name': 'rn50_python',
            'input_name': 'PYTHON_INPUT_0',
        }
    ]
    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    run_benchmark(model_descriptors, batch_sizes)
