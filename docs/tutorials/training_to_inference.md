# DALI - from training to inference

This tutorial presents the simple way of going from training to inference. We are focusing on a particular scenario: 
how to deploy a model, when it has been trained using DALI as a preprocessing tool. An example that constitutes this 
tutorial is the [EfficientNet](https://github.com/NVIDIA/DALI/tree/main/docs/examples/use_cases/pytorch/efficientnet) 
from DALI repository. This tutorial is split in two parts - [Theory](#Theory) and [Practice](#Practice). Lastly, while we cover only the 
deployment of DALI preprocessing graph, we are using handful of other NVIDIA tools useful for setting up the inference. 
Documentation and links to those tools will be provided, but the specifics of these tools won't be explained - should 
you like to know more details about them, please refer to the documentation.

## Assumptions

Since we are tackling a specific use-case in this tutorial, we will set up several assumptions:
1. We are deploying an EfficientNet model, trained with PyTorch. The checkpoint is saved using `torch.save` and DALI 
has been used as a preprocessing tool.
2. The input data to this model reflects the input data used during training. To be specific - we will infer on encoded 
JPEG files.
3. The training script contains DALI model, implemented using DALI's `@pipeline_def` decorator. Should you have the 
DALI pipeline implemented using other approach - please convert to `@pipeline_def` first.
4. Triton Inference Server will be used to orchestrate the inference.
5. The entire preprocessing part of the inference will happen on the server side, using the GPU.

## Theory

Unfortunately, it is not possible to introduce complete and precise algorithm for setting up the inference 
(if it would, we'd just put together a script for it). However, the following is a highly useful guide TODO
1. Adjust a DALI Pipeline for the inference:
    1. Make sure that the DALI pipeline you’ll be working on is the pipeline that contains the operations for the inference. Usually the **training** preprocessing and **inference** preprocessing differs - the former one contains some random operations that are used to augment the dataset. Most often the inference pipeline will match the **validation** pipeline (and not the training one).
    1.  Change all the input operators (except the Video operators) inside the DALI pipeline to the `fn.external_source` operator.
        1.  In the case of Video operators, use either `fn.inputs.video` or `fn.external_source + fn.decoders.video`, [depending on the input data properties](https://docs.nvidia.com/deeplearning/dali/main-user-guide/docs/operations/nvidia.dali.fn.experimental.inputs.video.html).
    1.  Add the `name` parameter to every input operator.
    1.  If you’d like to use the [Autoconfiguration](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_configuration.html#auto-generated-model-configuration) of DALI model:
        1. Add `ndim` parameter to every input operator,
        1. Add `dtype` parameter to every input operator,
        1. Add `output_ndim` parameter to the `@pipeline_def`,
        1. Add `output_dtype` parameter to the `@pipeline_def`.
1. Create a [Model Repository](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_repository.html).
    1. Set up your Deep Learning model. Description of this step is outside of the scope of this tutorial.
    1. Create a directory for the DALI model.
    1. Insert the DALI pipeline definition (prepared in the **Step 1**) into the `dali.py` file in the version directory.
    1. Create a configuration file (unnecessary, if you are using Model Autoconfiguration for DALI Backend).
    1. Combine the Deep Learning and DALI models using [model ensemble](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/architecture.html#ensemble-models) or [BLS script](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/index.html?highlight=business%20logic%20scripting).
1. Run the Triton server and send some requests to it.

Enough with the theory. Let's go to practice.

## Practice

### ➔ 1: Adjusting DALI Pipeline

As mentioned earlier, the validation pipeline usually reflects the inference preprocessing better. Here's one used in 
our EfficientNet example:
```python
@pipeline_def
def validation_pipe(data_dir, interpolation, image_size, image_crop, output_layout, rank=0,
                    world_size=1):
    jpegs, label = fn.readers.file(name="Reader", file_root=data_dir, shard_id=rank,
                                   num_shards=world_size, random_shuffle=False, pad_last_batch=True)

    images = fn.decoders.image(jpegs, device="mixed", output_type=types.RGB)

    images = fn.resize(images, resize_shorter=image_size, interp_type=interpolation,
                       antialias=False)

    output = fn.crop_mirror_normalize(images, dtype=types.FLOAT, output_layout=output_layout,
                                      crop=(image_crop, image_crop),
                                      mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                                      std=[0.229 * 255, 0.224 * 255, 0.225 * 255])
    return output, label
```
We're adjusting it by swapping the Reader with ExternalSource and adding a `name` parameter to it. Since in our example 
we will explicitly create a `config.pbtxt` file for DALI model, we're not adding `ndim` and `dtype` arguments to 
`fn.external_source`. We are, however, using [Autoserialization](https://github.com/triton-inference-server/dali_backend#autoserialization), 
hence the `@autoserialize` decorator on top.
```python
@autoserialize
@pipeline_def
def inference_pipe(interpolation, image_size, image_crop, output_layout):
    jpeg = fn.external_source(name="DALI_INPUT")

    images = fn.decoders.image(jpegs, device="mixed", output_type=types.RGB)

    images = fn.resize(images, resize_shorter=image_size, interp_type=interpolation,
                       antialias=False)

    output = fn.crop_mirror_normalize(images, dtype=types.FLOAT, output_layout=output_layout,
                                      crop=(image_crop, image_crop),
                                      mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                                      std=[0.229 * 255, 0.224 * 255, 0.225 * 255])
    return output
```
The snippet above shall be saved as `dali.py` file inside your model repository.
### ➔ 2 : Create a model repository
Putting together [model repository](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_repository.html) 
from the perspective of DALI is really simple. All you need to do is to save the DALI Pipeline inside `dali.py` file 
and put it inside corresponding model version directory.

However, for the real-life inference scenario, we also need to add a Deep Learning model to the equation. 
In this tutorial we will use the [DeepLearningExamples' Efficientnet model](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Classification/ConvNets/efficientnet) with the checkpoint 
downloaded from [TorchHub](https://github.com/NVIDIA/DeepLearningExamples/blob/master/hubconf.py), 
converted to TRT format. Model conversion is a broad topic and it can't be handled in this tutorial. 
Please refer to [Model Navigator documentation](https://triton-inference-server.github.io/model_navigator/) for details. 
Also, the [Efficientnet example](https://github.com/triton-inference-server/dali_backend/tree/main/docs/examples/efficientnet) 
in DALI Backend repository contains an [example script](https://github.com/triton-inference-server/dali_backend/blob/main/docs/examples/efficientnet/deploy_on_triton.py) 
that performs the conversion for our model.

Going back to our example, we are using the `efficientnet-b0` and the `preprocessing` models, with the latter containing
`dali.py` file defined above. To use these two together, we are creating the ensemble model `efficientnet_ensemble`. 
Below is the outline of the model repository in our example. You can find the [model configuration](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_configuration.html) 
files in the [Efficientnet example](https://github.com/triton-inference-server/dali_backend/tree/main/docs/examples/efficientnet/model_repository) 
in DALI Backend repository.
```
model_repository
├── efficientnet-b0
│   ├── 1
│   │   └── model.plan
│   └── config.pbtxt
├── efficientnet_ensemble
│   ├── 1
│   └── config.pbtxt
└── preprocessing
    ├── 1
    │   └── dali.py
    └── config.pbtxt
```

### ➔ 3 : Run Triton server and send requests
TODO

### Triton client hints
As mentioned earlier, you can find the examples of Triton client implementation inside [DALI Backend repository](https://github.com/triton-inference-server/dali_backend/tree/main/qa). 
The purpose of this tutorial is not to fully explain how to create a Triton client module. However, here are a couple of
suggestions when creating a Triton client specifically for DALI model.

#### Loading data as binary buffer
NVIDIA hardware offers image and video decoding acceleration using [Hardware JPEG Decoder](https://developer.nvidia.com/blog/leveraging-hardware-jpeg-decoder-and-nvjpeg-on-a100/) and [NVDEC cores](https://developer.nvidia.com/video-codec-sdk). 
Since DALI provides easy-access Python interface for both, you'd typically like to perform data decoding on the server side. 
Therefore, the Triton client will typically send encoded data to the server. We suggest using `np.fromfile` or similar 
approach to read the binary data from disk (this one prooved to be the fastest):
```python
def load_image(image_path):
    return np.fromfile(image_path, dtype=np.uint8)
```

#### Batching data in the request
With Triton and DALI Backend you can leverage batching in two ways. Firstly, you can turn on [Dynamic Batching](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_configuration.html#dynamic-batcher). 
With this feature, you send the samples one-by-one to the server and the server automatically puts together a batch of data for you. 
On the other hand, you may want to create a batch of data on the client side. Which brings us to the next suggestion...

#### Triton accepts batches with uniform shapes
When implementing the inference scenario that operates on batches of data, you shall remember that Triton accepts data 
with uniform shape. While this approach is something common in Deep Learning, when introducing data preprocessing the 
situation changes. Since image/video decoding would very likely be one of the operations in your preprocessing pipeline,
the input data from the client would comprise encoded streams. A batch with encoded JPEGs will unlikely have an uniform 
shape. The solution to this would be to pad the encoded samples with `0s`, [like in the `inception_ensemble` example](https://github.com/triton-inference-server/dali_backend/blob/075bb874ae99d20bf3bc67e26937cdb7b05a3b20/qa/L0_inception_ensemble/ensemble_client.py#L86).

When using Dynamic Batching, you are typically sending one sample per request and Triton Server put together a batch for you. 
However, you still need to keep in mind the non-uniform batch shape requirement. In this case you can leverage the [Ragged Batching](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/ragged_batching.html) 
feature.
