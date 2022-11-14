# Decoding video in DALI backend

## Splitting sequences into subsequences
Often the video model we want to run inference on expects sequences of some fixed lentght. In such case we may want to split the decoded video
into a batch of sub-sequences of that given length. We can achieve that utilizing the `fn.reshape` operator from DALI together with a model parameter
`"split_along_outer_axis"` set in configuration file. Lets see an example of using this combination.

```python
FRAMES_PER_SEQUENCE = 5
OUT_WIDTH = 300
OUT_HEIGHT = 300

@autoserialize
@dali.pipeline_def(batch_size=256, num_threads=min(mp.cpu_count(), 4), device_id=0,
                   output_dtype=dali.types.UINT8, output_ndim=[5, 4, 1])
def pipeline():
  vid = fn.external_source(device='cpu', name='INPUT', ndim=1, dtype=dali.types.UINT8)
  seq = fn.experimental.decoders.video(vid, device='mixed')
  seq = fn.resize(seq, resize_x=OUT_WIDTH, resize_y=OUT_HEIGHT)

  seq = fn.pad(seq, axis_names='F', align=FRAMES_PER_SEQUENCE)
  return fn.reshape(seq, shape=[-1, FRAMES_PER_SEQUENCE, OUT_HEIGHT, OUT_WIDTH, 3], name='OUTPUT')
```
You can see that we added two operators at the end of the pipeline - `pad` and `reshape`. Padding samples is needed because we want to split them into sub-sequneces of `FRAMES_PER_SEQUENCE` frames each - that means that we need to ensure that the total number of frames in a sample is divisible by `FRAMES_PER_SEQUENCE`. The result of this operator is adding frames filled with zeroes to the end of the sample to align to that number.

After padding the samples, we use the `reshape` operator to add another dimension to the data, so layout-wise the transformation could be represented as `FHWC` -> `NFHWC`, where `N` dimension represents index of a sub-sequence. An example flow could be described as follows: lets say a sample after resizing has a shape of `(13, 300, 300, 3)` - it's a sequence of 13 frames. Assuming the `FRAMES_PER_SEQUENCE` is 5, the sample is then padded to shape `(15, 300, 300, 3)`. Then, the reshape, without modifying any data, changes it to `(3, 5, 300, 300, 3)` - 3 sub-sequences with 5 frames each.

Reshaping data in a described way deosn't solve our problem completely but we can combine it with an additional model parameter that will split the output tensors and treat each sub-sequence as a separate sample in the batch.

The parameter is called `split_along_outer_axis` and can be set in the configuration file as follows:

```pbtxt
parameters [
  {
    key: "split_along_outer_axis",
    value: { string_value: "OUTPUT" }
  }
]
```
The value of the parameter (specified as `string_value`) is an output name (or list of names seperated with a colon) that should be transformed. We can see the effect of this parameter on the following example: lets assume the DALI pipeline produced a batch of two samples with the shapes {(1, 5, 300, 300, 3), (2, 5, 300, 300, 3)}, so a sample with 1 sub-sequence and a sample with 2 sub-sequences. The batch produced by our model will then have shape {(5, 300, 300, 3), (5, 300, 300, 3), (5, 300, 300, 3)} - a batch of 3 sequences.

As you can see, the said parameter helps us also handle video files of different lenghts. Outputting such sequences by the DALI pipeline is not feasible in Triton because batches in Triton need to have uniform shape (every sample in the batch should have the same shape). Using the `reshape` - `"split_along_outer_axis"` combination can transform unevenly shaped sequences into a batch of evenly shaped sub-sequences.
