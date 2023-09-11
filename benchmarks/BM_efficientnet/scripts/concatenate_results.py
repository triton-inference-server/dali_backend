# The MIT License (MIT)
#
# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES
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
import os
import re
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--reports-path', type=str, required=True, help='Path to the directory with the reports')
    return parser.parse_args()


def list_reports(reports_path):
    return os.listdir(reports_path)


def parse_batch_size(report_file_name):
    return int(re.search(r'\d+', report_file_name).group())


def validate_report_file_name(report_file_name):
    return re.match(r'report-\d+\.csv', report_file_name) is not None


def insert_batch_size_column(report_csv, batch_size):
    report_csv.insert(0, 'Batch size', batch_size)


def main(reports_path, result_path):
    reports_list = list_reports(reports_path)
    assert all(validate_report_file_name(rfn) for rfn in reports_list)
    reports = []
    for rep in reports_list:
        report = pd.read_csv(os.path.join(reports_path, rep))
        insert_batch_size_column(report, parse_batch_size(rep))
        reports.append(report)
    result = pd.concat(reports)
    result.to_csv(result_path, sep=',')


if __name__ == '__main__':
    args = parse_args()
    main(args.reports_path, f'{args.reports_path}/combined.csv')
