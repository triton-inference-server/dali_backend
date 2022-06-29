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

## Autoserialization

This example present the autoserialization feature in DALI Backend. The `dali.py` file contains a definition of DALI Pipeline. In general, user would need to serialize this pipeline to `model.dali` file. However, leveraging the autoserialization, DALI Backend will serialize the model itself. For more details about autoserialization, please refer to `@autoserialize` documentation.

## Run the example

To run the example, you have to download InceptionV3 model into the model repo. Please refer to `setup_inception_example.sh` for details, how to do this.

## Remember

As always in DALI Backend case, remember that `dali.fn.external_source`'s `name` parameter must match
with input name provided in `config.pbtxt` file.
