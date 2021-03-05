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

import numpy as np

def batcher(dataset, batch_size, n_iterations=-1):
    """
    Generator, that splits dataset into batches with given batch size

    :param dataset: list of ndarrays, where every ndarray is one sample
    :param batch_size: Requested batch size
    :param n_iterations: Requested number of iterations. Samples gathered in ``dataset`` argument
                         will be used in round-robin manner to form the proper number of batches.
                         Default value (-1) means, that number of iterations will be calculated
                         using the formula: ``n_iterations = len(dataset) // batch_size``
    :return: Yields batches
    """
    n_batches = len(dataset) // batch_size if n_iterations == -1 else n_iterations
    dataset_idx = 0
    for _ in range(n_batches):
        if dataset_idx + batch_size < len(dataset):
            yield dataset[dataset_idx: dataset_idx + batch_size]
        else:
            yield np.concatenate((np.repeat(dataset, repeats=batch_size // len(dataset), axis=0),
                                  dataset[:batch_size % len(dataset)]))
        dataset_idx += batch_size
