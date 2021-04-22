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
import inspect


def _select_batch_size(batch_size_provider, batch_idx):
    if inspect.isgenerator(batch_size_provider):
        return next(batch_size_provider)
    elif isinstance(batch_size_provider, list):
        return batch_size_provider[batch_idx % len(batch_size_provider)]
    elif isinstance(batch_size_provider, int):
        return batch_size_provider
    raise TypeError("Incorrect batch_size_provider type. Actual: ", type(batch_size_provider))


def batcher(dataset, batch_size_provider, n_iterations=-1):
    """
    Generator, that splits dataset into batches with given batch size

    :param dataset: ndarray, where the outermost dimension is the size of the dataset. I.e.
                    ``dataset[0]`` is the first sample, ``dataset[1]`` is the second etc.
    :param batch_size_provider: Provides sizes for every batch.
                                * If the argument is a scalar, every batch will have the same size.
                                * If a list, sizes will be picked consecutively
                                (round-robin if necessary).
                                * If a generator, every batch size will be determined by a
                                ``next(batch_size_provider)`` call.
    :param n_iterations: Requested number of iterations. Samples gathered in ``dataset`` argument
                         will be used in round-robin manner to form the proper number of batches.
                         Default value (-1) means, that number of iterations will be inferred from
                         the dataset size - every sample will be used at most one time and last
                         (incomplete) batch will be dropped.
    :return: Yields batches
    """
    iter_idx = 0
    curr_sample = 0
    while True:
        try:
            batch_size = _select_batch_size(batch_size_provider, iter_idx)
        except StopIteration:
            return
        dataset_size = dataset.shape[0]

        # Stop condition
        if n_iterations == -1:
            if curr_sample + batch_size >= dataset_size:
                return
        else:
            if iter_idx >= n_iterations:
                return

        if curr_sample + batch_size < dataset_size:
            yield dataset[curr_sample: curr_sample + batch_size]
        else:
            # Get as many samples from this revolution of the dataset as possible,
            # then repeat the dataset as many revolutions as needed
            # and finally take the remaining samples from the last revolution
            suffix = dataset_size - curr_sample
            n_rep = (batch_size - suffix) // dataset_size
            prefix = batch_size - (suffix + dataset_size * n_rep)
            yield np.concatenate(
                (dataset[curr_sample:],
                 np.repeat(dataset, repeats=n_rep, axis=0),
                 dataset[:prefix])
            )
        curr_sample = (curr_sample + batch_size) % dataset_size
        iter_idx += 1
