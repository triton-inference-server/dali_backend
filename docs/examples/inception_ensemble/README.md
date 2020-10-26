# InceptionV3 inference using DALI

This is the example, that visualizes InceptionV3 inference, where DALI
is used for data preprocessing.

Our Inception model accepts images as an input. These images have to be
resized to precisely `[299, 299]` resolution and they have to be normalized.
Thanks to DALI you can implement these operations easily (using DALI's python API)
and GPU-accelerate them at the same time.

## The ensemble

Triton uses model ensembling to put together multiple models into one
bulk inferring. This example is such an ensemble, consisting of two
models: DALI model for image preprocessing and InceptionV3 model for
actual inference.  For more info on ensembling, refer to [Triton
docs](https://github.com/triton-inference-server/server/blob/master/docs/architecture.md#ensemble-models).

## Run the example

To run the example, please follow these simple steps:
1. Download InceptionV3 model into the model repo
1. Serialize DALI pipeline and put it inside model repo.

Both of the step above are presented in the `setup_inception_example.sh`

## Remember

As always in DALI Backend case, remember that `dali.fn.external_source`'s `name` parameter must match
with input name provided in `config.pbtxt` file.
