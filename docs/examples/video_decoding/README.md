# Decoding video in DALI backend

## Splitting sequences into subsequences
Often, the video model we want to run inference on expects the sequences of some fixed length. In such case we may want to split the decoded video
into a batch of sub-sequences of that given length. We can achieve that utilizing the `fn.pad` and `fn.reshape` operators from DALI together with a model parameter
`"split_along_outer_axis"` set in the model configuration file. Let's see an example of using this combination.

```python
FRAMES_PER_SEQUENCE = 5
OUT_WIDTH = 300
OUT_HEIGHT = 300

@autoserialize
@dali.pipeline_def(batch_size=256, num_threads=4, device_id=0,
                   output_dtype=dali.types.UINT8, output_ndim=5)
def pipeline():
  vid = fn.external_source(device='cpu', name='INPUT', ndim=1, dtype=dali.types.UINT8)
  seq = fn.experimental.decoders.video(vid, device='mixed')
  seq = fn.resize(seq, resize_x=OUT_WIDTH, resize_y=OUT_HEIGHT)

  seq = fn.pad(seq, axis_names='F', align=FRAMES_PER_SEQUENCE)
  return fn.reshape(seq, shape=[-1, FRAMES_PER_SEQUENCE, OUT_HEIGHT, OUT_WIDTH, 3], name='OUTPUT')
```
You can see that we added two operators at the end of the pipeline - `fn.pad` and `fn.reshape` operators. Since we want to split the sequences into sub-sequneces of `FRAMES_PER_SEQUENCE` frames each, we must ensure that the total number of frames in every sample is divisible by `FRAMES_PER_SEQUENCE`. This is achieved with the `fn.pad` operator which adds frames filled with zeroes to the end of each sample to align to that length.

After padding the samples, we use the `fn.reshape` operator to add another dimension to the data, so layout-wise the transformation could be represented as `FHWC` -> `NFHWC`, where `N` dimension represents index of a sub-sequence. An example flow could be described as follows:
```python
  seq = fn.resize(seq, resize_x=OUT_WIDTH, resize_y=OUT_HEIGHT)
  # Let's assume we have a sequence of 13 frames. Shape: (13, 300, 300, 3).
  seq = fn.pad(seq, axis_names='F', align=FRAMES_PER_SEQUENCE)
  # With FRAMES_PER_SEQUENCE equal to 5, the sample will be padded to shape (15, 300, 300, 3).
  return fn.reshape(seq, shape=[-1, FRAMES_PER_SEQUENCE, OUT_HEIGHT, OUT_WIDTH, 3], name='OUTPUT')
  # The output of a pipeline has then shape of (3, 5, 300, 300, 3) - 3 sub-sequences with 5 frames each.
```

Reshaping data in a described way doesn't solve our problem completely because we are still left with samples containing different numbers of sub-sequences. We can use an additional model parameter to split the output tensors and treat each sub-sequence as a separate sample in the batch.

The parameter is called `split_along_outer_axis` and can be set in the model configuration file as follows:

```pbtxt
parameters [
  {
    key: "split_along_outer_axis",
    value: { string_value: "OUTPUT" }
  }
]
```
The value of the parameter (specified as `string_value`) is the name of the output (or list of names separated with a colon) that should be transformed. We can see the effect of this parameter on the following example: let's assume the DALI pipeline produced a batch of two samples with the shapes `{(1, 5, 300, 300, 3), (2, 5, 300, 300, 3)}`, so the first sample with 1 sub-sequence and the seconds sample with 2 sub-sequences. The batch produced by our model will then have shape `{(5, 300, 300, 3), (5, 300, 300, 3), (5, 300, 300, 3)}` - a batch of 3 sequences.

As you can see, the said parameter helps us also handle video files of different lenghts. Outputting such sequences by the DALI pipeline is not feasible in Triton because batches in Triton need to have a uniform shape (every sample in the batch should have the same shape). Using the `fn.reshape` + `"split_along_outer_axis"` combination can transform unevenly shaped sequences into a batch of evenly shaped sub-sequences.
