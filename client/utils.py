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


def _select_batch_size(batch_size_provider, batch_idx):
    if callable(batch_size_provider):
        return batch_size_provider()
    elif isinstance(batch_size_provider, list):
        return batch_size_provider[batch_idx % len(batch_size_provider)]
    elif isinstance(batch_size_provider, int):
        return batch_size_provider
    raise TypeError("Incorrect batch_size_provider type. Actual: ", type(batch_size_provider))


def batcher(dataset, batch_size_provider, n_iterations=-1):
    """
    Generator, that splits dataset into batches with given batch size

    :param dataset: list of ndarrays, where every ndarray is one sample
    :param batch_size_provider: Provides sizes for every batch.
                                If the argument is a scalar, every batch will have the same size.
                                If a list, sizes will be picked consecutively
                                (round-robin if necessary).
                                If a callable, every iteration will call ``batch_size_provider``
                                to obtain the size.
    :param n_iterations: Requested number of iterations. Samples gathered in ``dataset`` argument
                         will be used in round-robin manner to form the proper number of batches.
                         Default value (-1) means, that number of iterations will be inferred from
                         the dataset size - every sample will be used at most one time and last
                         (incomplete) batch will be dropped.
    :return: Yields batches
    """
    iter_idx = 0
    curr_sample = 0

    if n_iterations == -1:
        while True:
            batch_size = _select_batch_size(batch_size_provider, iter_idx)
            if curr_sample + batch_size > len(dataset):
                return
            yield dataset[curr_sample: curr_sample + batch_size]
            curr_sample += batch_size
            iter_idx += 1
    else:
        while True:
            batch_size = _select_batch_size(batch_size_provider, iter_idx)
            print(batch_size)
            if iter_idx >= n_iterations:
                return
            elif curr_sample + batch_size < len(dataset):
                yield dataset[curr_sample: curr_sample + batch_size]
            else:
                suffix = len(dataset) - curr_sample
                n_rep = (batch_size - suffix) // len(dataset)
                prefix = batch_size - (suffix + len(dataset) * n_rep)
                yield np.concatenate(
                    (dataset[curr_sample:],
                     np.repeat(dataset, repeats=n_rep, axis=0),
                     dataset[:prefix])
                )
            curr_sample = (curr_sample + batch_size) % len(dataset)
            iter_idx += 1



def gen():
    global j
    j = 2 if j == 1 else 1
    return j


def main():
    dataset = np.array([i for i in range(5)])
    for b in batcher(dataset, [3], 4):
        print(b)


if __name__ == '__main__':
    main()

