# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#           http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import itertools
import math
import multiprocessing

import numpy as np

import nvidia.dali as dali
import nvidia.dali.fn as fn
import nvidia.dali.types as types


def _interleave_lists(*lists):
    """
    [*, **, ***], [1, 2, 3], [a, b, c] -> [*, 1, a, **, 2, b, ***, 3, c]
    Returns:
        iterator over interleaved list
    """
    assert all(
        (len(lists[0]) == len(test_l) for test_l in lists)
    ), "All lists have to have the same length"
    return itertools.chain(*zip(*lists))


def _tuples2list(tuples: list):
    """
    [(a, b), (c, d)] -> [[a, c], [b, d]]
    """
    return map(list, zip(*tuples))


@dali.pipeline_def
def dali_asr_pipeline(
        train_pipeline,  # True if training, False if validation
        file_root,
        file_list,
        sample_rate,
        silence_threshold,
        resample_range,
        discrete_resample_range,
        window_size,
        window_stride,
        nfeatures,
        nfft,
        frame_splicing_factor,
        dither_coeff,
        pad_align,
        preemph_coeff,
        do_spectrogram_masking=False,
        cutouts_generator=None,
        shard_id=0,
        n_shards=1,
        preprocessing_device="gpu",
        is_triton_pipeline=False,
):
    do_remove_silence = silence_threshold is not None

    def _div_ceil(dividend, divisor):
        return (dividend + (divisor - 1)) // divisor

    if is_triton_pipeline:
        assert not train_pipeline, "Pipeline for Triton shall be a validation pipeline"
        encoded = fn.external_source(device="cpu", name="DALI_INPUT_0", no_copy=True)
    else:
        encoded, label = fn.readers.file(
            device="cpu",
            name="file_reader",
            file_root=file_root,
            file_list=file_list,
            shard_id=shard_id,
            num_shards=n_shards,
            shuffle_after_epoch=train_pipeline,
        )

    speed_perturbation_coeffs = None
    if resample_range is not None:
        if discrete_resample_range:
            values = [resample_range[0], 1.0, resample_range[1]]
            speed_perturbation_coeffs = fn.random.uniform(device="cpu", values=values)
        else:
            speed_perturbation_coeffs = fn.random.uniform(
                device="cpu", range=resample_range
            )

    if train_pipeline and speed_perturbation_coeffs is not None:
        dec_sample_rate_arg = speed_perturbation_coeffs * sample_rate
    elif resample_range is None:
        dec_sample_rate_arg = sample_rate
    else:
        dec_sample_rate_arg = None

    audio, _ = fn.decoders.audio(
        encoded, sample_rate=dec_sample_rate_arg, dtype=types.FLOAT, downmix=True
    )
    if do_remove_silence:
        begin, length = fn.nonsilent_region(audio, cutoff_db=silence_threshold)
        audio = fn.slice(audio, begin, length, axes=[0])

    # Max duration drop is performed at DataLayer stage

    if preprocessing_device == "gpu":
        audio = audio.gpu()

    if dither_coeff != 0.0:
        audio = audio + fn.random.normal(device=preprocessing_device) * dither_coeff

    audio = fn.preemphasis_filter(audio, preemph_coeff=preemph_coeff)

    spec = fn.spectrogram(
        audio,
        nfft=nfft,
        window_length=window_size * sample_rate,
        window_step=window_stride * sample_rate,
    )

    mel_spec = fn.mel_filter_bank(
        spec, sample_rate=sample_rate, nfilter=nfeatures, normalize=True
    )

    log_features = fn.to_decibels(
        mel_spec, multiplier=np.log(10), reference=1.0, cutoff_db=math.log(1e-20)
    )

    log_features_len = fn.shapes(log_features)
    if frame_splicing_factor != 1:
        log_features_len = _div_ceil(log_features_len, frame_splicing_factor)

    log_features = fn.normalize(log_features, axes=[1])
    log_features = fn.pad(log_features, axes=[1], fill_value=0, align=pad_align, shape=(-1,))

    if train_pipeline and do_spectrogram_masking:
        anchors, shapes = fn.external_source(
            source=cutouts_generator, num_outputs=2, cycle=True
        )
        log_features = fn.erase(
            log_features,
            anchor=anchors,
            shape=shapes,
            axes=[0, 1],
            fill_value=0,
            normalized_anchor=True,
        )

    # When modifying DALI pipeline returns, make sure you update `output_map`
    # in DALIGenericIterator invocation
    if not is_triton_pipeline:
        return log_features.gpu(), label.gpu(), log_features_len.gpu()
    else:
        return fn.cast(log_features.gpu(), dtype=types.FLOAT16)


def serialize_dali_pipeline(filepath):
    p = dali_asr_pipeline(
        train_pipeline=False,
        file_root=None,
        file_list=None,
        sample_rate=16000,
        silence_threshold=-60,
        resample_range=None,
        discrete_resample_range=None,
        window_size=.02,
        window_stride=.01,
        nfeatures=64,
        nfft=512,
        frame_splicing_factor=1,
        dither_coeff=1e-5,
        pad_align=16,
        preemph_coeff=.97,
        preprocessing_device="gpu",
        is_triton_pipeline=True,
        batch_size=1,
        num_threads=16,
        device_id=0,
    )
    p.serialize(filename=filepath)


if __name__ == '__main__':
    serialize_dali_pipeline('1/model.dali')
