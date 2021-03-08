# Triton Inference Server with DALI backend

This repo is an example of [DALI (Data Loading Library) backend](https://github.com/NVIDIA/DALI) for image classification on [Triton Inference server](https://github.com/triton-inference-server/server).

ResNet50 model optimized by [TensorRT](https://developer.nvidia.com/tensorrt) is used for image classification. 


#### Dependencies

* Export ONNX and build TensorRT
  * nvcr.io/nvidia/pytorch:20.12-py3
* Triton Inference Server for DALI backend
  * nvcr.io/nvidia/tritonserver:20.12-py3
  * https://github.com/triton-inference-server/dali_backend

* Client
  * nvcr.io/nvidia/tritonserver:20.10-py3-clients

#### Setting up the ONNX-TensorRT and DALI backend ENV

```
$git clone https://github.com/triton-inference-server/dali_backend --recursive
$cd dali_backend
$docker build -t triton_dali_backend -f Dockerfile .
$cd docs/examples/resnet50_trt
```

#### Ready for ResNet50-TensorRT model and DALI pipeline

Run `setup_resnet50_trt_example.sh` for building resnet50-trt model and DALI pipeline.
```
$bash setup_resnet50_trt_example.sh
```

If you don't want to run the script, follow 1-3 steps below.

##### 1.  Converting PyTorch Model to ONNX-model 

Create directories for model repository and run `onnx_exporter.py` for conversion using PyTorch model to ONNX model. In this case, ResNet50 model is converted to ONNX format. `width` and `height` dims are fixed at 224 but dynamic axes arguments for dynamic batch is used. 

```
$mkdir -p model_repository/dali/1
$mkdir -p model_repository/ensemble_dali_resnet50/1
$mkdir -p model_repository/resnet50_trt/1

$docker run -it --gpus=all -v $(pwd):/workspace nvcr.io/nvidia/pytorch:20.12-py3 bash
$python onnx_exporter.py --save model.onnx
```

##### 2. Building ONNX-model to TensorRT engine

Set the arguments for enabling fp16 precision `--fp16` and for dynamic shapes using a profile `--minShapes`, `--optShapes`, and `maxShapes` with `--explicitBatch`

```python
$trtexec --onnx=model.onnx --saveEngine=./model_repository/resnet50_trt/1/model.plan --explicitBatch --minShapes=input:1x3x224x224 --optShapes=input:1x3x224x224 --maxShapes=input:256x3x224x224 --fp16
```

##### 3. Serialize DALI pipeline 

Run `serialize_dali_pipeline.py` for generating  DALI pipeline. 

```
$python serialize_dali_pipeline.py --save ./model_repository/dali/1/model.dali
```

If had a dedicated hardware decoder, the hardware is available. Set the preprocessing pipeline  `resize` and `normalize` using `dali.pipeline.Pipeline` and serialize pipeline using `Pipeline.serialize`

````python
pipe = dali.pipeline.Pipeline(batch_size=256, num_threads=4, device_id=0)
with pipe:
    images = dali.fn.external_source(device="cpu", name="DALI_INPUT_0")
    images = dali.fn.image_decoder(images, device="mixed", output_type=types.RGB)
    images = dali.fn.resize(images, resize_x=224, resize_y=224)
    images = dali.fn.crop_mirror_normalize(images,
                                           dtype=types.FLOAT,
                                           output_layout="CHW",
                                           crop=(224, 224),
                                           mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                                           std=[0.229 * 255, 0.224 * 255, 0.225 * 255])


    pipe.set_outputs(images)
    pipe.serialize(filename=args.save)
    
````



#### Run Triton Inference Server

![](./images/ensemble.PNG)

```bash
model_repository
├── dali
│   ├── 1
│   │   └── model.dali
│   └── config.pbtxt
├── ensemble_dali_resnet50
│   ├── 1
│   └── config.pbtxt
└── resnet50_trt
    ├── 1
    │   └── model.plan
    ├── config.pbtxt
    └── labels.txt
```

For preprocessing using DALI 

```
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
    dims: [ 3, 224, 224 ]
}
]
```

For ResNet50 model using TensorRT,

```
name: "resnet50_trt"
platform: "tensorrt_plan"
max_batch_size: 256
input [
{
    name: "input"
    data_type: TYPE_FP32
    dims: [ 3, -1, -1 ]
    
}
]
output[
{
    name: "output"
    data_type: TYPE_FP32
    dims: [ 1000 ]
    label_filename: "labels.txt"
}
]

```

For ensemble model for image classification pipeline, 

```
name: "ensemble_dali_resnet50"
platform: "ensemble"
max_batch_size: 256
input [
  {
    name: "INPUT"
    data_type: TYPE_UINT8
    dims: [ -1 ]
  }
]
output [
  {
    name: "OUTPUT"
    data_type: TYPE_FP32
    dims: [ 1000 ]
  }
]
ensemble_scheduling {
  step [
    {
      model_name: "dali"
      model_version: -1
      input_map {
        key: "DALI_INPUT_0"
        value: "INPUT"
      }
      output_map {
        key: "DALI_OUTPUT_0"
        value: "preprocessed_image"
      }
    },
    {
      model_name: "resnet50_trt"
      model_version: -1
      input_map {
        key: "input"
        value: "preprocessed_image"
      }
      output_map {
        key: "output"
        value: "OUTPUT"
      }
    }
  ]
}
```

Run Triton inference server

```
$docker run --gpus=all --rm -p8000:8000 -p8001:8001 -p8002:8002 -v$(pwd):/workspace/ -v/$(pwd)/model_repository:/models triton_dali_backend tritonserver --model-repository=/models
```

#### Request image classification

Create gRPC client via URL

```python
triton_client = tritongrpcclient.InferenceServerClient(url=args.url, verbose=False)
```

Load raw image from `numpy` and configurate input and output with the name, shape and datatype. 

```
inputs = []
outputs = []
input_name = "INPUT"
output_name = "OUTPUT"
image_data = load_image(args.image)
image_data = np.expand_dims(image_data, axis=0)

inputs.append(tritongrpcclient.InferInput(input_name, image_data.shape, "UINT8"))
outputs.append(tritongrpcclient.InferRequestedOutput(output_name))

inputs[0].set_data_from_numpy(image_data)
```

Request inference and respond the results

```python
results = triton_client.infer(model_name=args.model_name,
                                    inputs=inputs,
                                    outputs=outputs)
output0_data = results.as_numpy(output_name)
```

Run `client.py` with the path to image `--image`

```
$docker run --rm --net=host -v $(pwd):/workspace/ nvcr.io/nvidia/tritonserver:20.10-py3-clientsdk python client.py --image <path to image> 
```

```bash
$wget https://raw.githubusercontent.com/triton-inference-server/server/master/qa/images/mug.jpg -O "mug.jpg"
$docker run --rm --net=host -v $(pwd):/workspace/ nvcr.io/nvidia/tritonserver:20.10-py3-clientsdk python client.py --image mug.jpg 
0.02642226219177246ms class:COFFEE MUG
```

