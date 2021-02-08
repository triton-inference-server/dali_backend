##### **NOTE:** `dali_backend` is available in `tritonserver-20.11` and later

# DALI TRITON Backend

This repository contains code for DALI Backend for Triton Inference Server.

![alt text](https://developer.nvidia.com/sites/default/files/akamai/dali.png)

**NVIDIA DALI (R)**, the Data Loading Library, is a collection of highly optimized building blocks, 
and an execution engine, to accelerate the pre-processing of the input data for deep learning applications.
DALI provides both the performance and the flexibility to accelerate different data pipelines as one library.
This library can then be easily integrated into different deep learning training and inference applications,
regardless of used deep learning framework.

To find out more about DALI please refer to our [main page](https://developer.nvidia.com/DALI).
[Getting started](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/examples/getting%20started.html#Getting-started)
and [Tutorials](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/examples/index.html)
will guide you through your first steps and [Supported operations](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/supported_ops.html)
will help you put together GPU-powered data processing pipelines.

## See any bugs?
Feel free to post an issue here or in DALI's [github repository](https://github.com/NVIDIA/DALI).

## How to use?

1. DALI data pipeline is expressed within Triton as a
[Model](https://github.com/triton-inference-server/server/blob/master/docs/architecture.md#models-and-schedulers).
To create such Model, you have to put together a [DALI
Pipeline](https://docs.nvidia.com/deeplearning/dali/master-user-guide/docs/examples/getting%20started.html#Pipeline)
in Python, and call the
[Pipeline.serialize](https://docs.nvidia.com/deeplearning/dali/master-user-guide/docs/pipeline.html#nvidia.dali.pipeline.Pipeline.serialize)
method to generate a Model file. As an example, we'll use simple
resizing pipeline:

        import nvidia.dali as dali
        
        pipe = dali.pipeline.Pipeline(batch_size=256, num_threads=4, device_id=0)
        with pipe:
            images = dali.fn.external_source(device="cpu", name="DALI_INPUT_0")
            images = dali.fn.image_decoder(images, device="mixed")
            images = dali.fn.resize(images, resize_x=224, resize_y=224)
            pipe.set_outputs(images)
            
        pipe.serialize(filename="/my/model/repository/path/dali/1/model.dali")

1. Model file shall be incorporated in Triton's [Model
Repository](https://github.com/triton-inference-server/server/blob/master/docs/model_repository.md).
Here's the example:
    
        model_repository
        └── dali
            ├── 1
            │   └── model.dali
            └── config.pbtxt

1. As it's typical in Triton, your DALI Model file shall be named `model.dali`.
You can override this name in the model configuration, by setting `default_model_filename` option.
Here's the whole `config.pbtxt` we use for the `ResizePipeline` example:

        name: "dali"
        backend: "dali"
        max_batch_size: 256
        input [
        {
            name: "DALI_INPUT_0"
            data_type: TYPE_UINT8
            dims: [ -1 ]
        }
        ]
        
        output [
        {
            name: "DALI_OUTPUT_0"
            data_type: TYPE_FP32
            dims: [ 224, 224, 3 ]
        }
        ]
        
## Tips & Tricks:
1. There's a high chance, that you'll want to use the `ops.ExternalSource` operator to feed the encoded 
images into DALI (or any other data for that matter).
1. Give your `ExternalSource` operator the same name you give to the Input in `config.pbtxt`

## Known limitations:
1. DALI's `ImageDecoder` accepts data only from the CPU - keep this in mind when putting together your DALI pipeline.
1. Triton accepts only homogeneous batch shape. Feel free to pad your batch of encoded images with zeros
1. Due to DALI limitations, you might observe unnaturally increased memory consumption when
defining instance group for DALI model with higher `count` than 1. We suggest using default instance
group for DALI model.


## How to build?

### Docker build
Building DALI Backend with docker is as simple as:

    git clone --recursive https://github.com/triton-inference-server/dali_backend.git
    cd dali_backend 
    docker build .

### Bare metal
#### Prerequisites
To build `dali_backend` you'll need `CMake 3.17+`
#### Using fresh DALI release
On the event you'd need to use newer DALI version than it's provided in `tritonserver` image,
you can use DALI's [nightly builds](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/installation.html#nightly-and-weekly-release-channels).
Just install whatever DALI version you like using pip (refer to the link for more info how to do it). 
In this case, while building `dali_backend`, you'd need to pass `-D TRITON_SKIP_DALI_DOWNLOAD=ON` 
option to your CMake build. `dali_backend` will find the latest DALI installed in your system and
use this particular version.
#### Building
Building DALI Backend is really straightforward. One thing to remember is to clone
`dali_backend` repository with all the submodules:

    git clone --recursive https://github.com/triton-inference-server/dali_backend.git
    cd dali_backend
    mkdir build
    cd build
    cmake ..
    make
    
The building process will generate `unittest` executable.
You can use it to run unit tests for DALI Backend


