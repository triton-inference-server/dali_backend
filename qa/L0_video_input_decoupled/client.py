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


from functools import partial
from itertools import cycle
import numpy as np
import queue
from os import environ
from glob import glob
import argparse

from tritonclient.utils import *
import tritonclient.grpc as t_client

import nvidia.dali.fn as fn
from nvidia.dali import pipeline_def


class UserData:

    def __init__(self):
        self._completed_requests = queue.Queue()


def callback(user_data, result, error):
    if error:
        user_data._completed_requests.put(error)
    else:
        user_data._completed_requests.put(result)


def get_dali_extra_path():
    return environ['DALI_EXTRA_PATH']


def input_gen():
    filenames = glob(f'{get_dali_extra_path()}/db/video/[cv]fr/*.mp4')
    filenames = filter(lambda filename: 'mpeg4' not in filename, filenames)
    filenames = filter(lambda filename: 'hevc' not in filename, filenames)
    for filename in filenames:
        print(filename)
        yield np.fromfile(filename, dtype=np.uint8)


FRAMES_PER_SEQUENCE = 5
BATCH_SIZE = 3
FRAMES_PER_BATCH = FRAMES_PER_SEQUENCE * BATCH_SIZE
N_FRAMES = 50

user_data = UserData()


@pipeline_def(batch_size=3, num_threads=3, device_id=0, prefetch_queue_depth=1)
def ref_pipeline(device):
    return fn.experimental.inputs.video(sequence_length=FRAMES_PER_SEQUENCE, name='data',
                                        device='mixed' if device == 'gpu' else 'cpu', last_sequence_policy='pad')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-u', '--url', type=str, required=False, default='localhost:8001',
                        help='Inference server GRPC URL. Default is localhost:8001.')
    parser.add_argument('-d', '--device', type=str, required=False, default='cpu', help='cpu or gpu')
    parser.add_argument('-n', '--n_iters', type=int, required=False, default=1, help='Number of iterations')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    model_name = 'model.dali' if args.device == 'cpu' else 'model_gpu.dali'
    with t_client.InferenceServerClient(url=args.url) as triton_client:
        triton_client.start_stream(callback=partial(callback, user_data))

        for req_id, input_data in zip(range(args.n_iters), cycle(input_gen())):
            inp = t_client.InferInput('INPUT', [1, input_data.shape[0]], 'UINT8')
            inp.set_data_from_numpy(input_data.reshape((1, -1)))

            outp = t_client.InferRequestedOutput('OUTPUT')

            request_id = str(req_id)
            triton_client.async_stream_infer(model_name=model_name,
                                             inputs=[inp],
                                             request_id=request_id,
                                             outputs=[outp])

            ref_pipe = ref_pipeline(device=args.device)
            ref_pipe.build()
            ref_pipe.feed_input('data', [input_data])

            recv_count = 0
            expected_count = (N_FRAMES + FRAMES_PER_BATCH - 1) // FRAMES_PER_BATCH
            result_dict = {}
            expected_dict = {}
            while recv_count < expected_count:
                data_item = user_data._completed_requests.get()
                if type(data_item) == InferenceServerException:
                    raise data_item
                else:
                    this_id = data_item.get_response().id
                    if this_id not in result_dict.keys():
                        result_dict[this_id] = []
                    result_dict[this_id].append(data_item)

                expected_result, = ref_pipe.run()
                if args.device == 'gpu':
                    expected_result = expected_result.as_cpu().as_array()
                else:
                    expected_result = expected_result.as_array()
                if this_id not in expected_dict.keys():
                    expected_dict[this_id] = []
                expected_dict[this_id].append(expected_result)

                recv_count += 1

            result_list = result_dict[request_id]
            expected_list = expected_dict[request_id]
            for res, exp in zip(result_list, expected_list):
                res = res.as_numpy("OUTPUT")
                if not np.allclose(res, exp):
                    import cv2
                    for i in range(res.shape[0]):
                        for j in range(res.shape[1]):
                            cv2.imwrite(f'res{i}.{j}.png', res[i][j])
                            cv2.imwrite(f'exp{i}.{j}.png', exp[i][j])
                    assert False, f"Results do not match: {res=} vs {exp=}. Shapes: {res.shape=} vs {exp.shape=}"

            print(f'ITER {req_id}: OK')
