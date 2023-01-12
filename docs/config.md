# DALI model configuration file

## Model parameters

### `split_along_outer_axis`
This parameters is used to split samples produced by a DALI pipeline removing their outer dimension.
Example use:
```pbtxt
parameters [
  {
    key: "split_along_outer_axis",
    value: { string_value: "OUTPUT1:OUTPUT2" }
  }
]
```
The value of the this parameter is a colon-separated list of output names that will be transformed.

Effect of the parameter can be described as follows. Let's assume that DALI pipeline produced a batch of _N_ _D+1_-dimensional tensors of following shapes:

$$\lbrace (n_1, d_1, ... d_D), ... (n_N, d_1, ... d_D)\rbrace$$

With the parameter set for this output, the batch produced by the model will uniformly shaped  _D_-dimensional samples, with batch-size equal to:

$$\sum_{i=1}^N n_i$$

with each sample having a shape of:

$$(d_1, ... d_D)$$

The option can be used to split batch of sequences into a batch of images or sub-sequences.
