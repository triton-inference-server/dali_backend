# perf_analyzer example with DALI image decoder


## Run the example

### tritonserver

To run the tritonserver, first you need to serialize a DALI pipeline and put it inside the model repository.

`setup_perf_analyzer_example.sh` is a convenience script that automatizes setting up.
Provided you have DALI installed in your system, you can just call `sh setup_perf_analyzer_example.sh`

When you have your model repository set up, you can run `tritonserver`. Be sure to replace `<path to model repo>` with the actual path in your system:

    docker run -it --rm --shm-size=1g --ulimit memlock=-1 --gpus all --ulimit stack=67108864 -p8000:8000 -p8001:8001 -p8002:8002 -v <path to model repo>:/models nvcr.io/nvidia/tritonserver:21.02-py3 tritonserver --model-repository=/models

### perf_analyzer

1. Pick an image for testing
1. Run Triton's client docker container:
 
    ```
    docker run -it --net=host nvcr.io/nvidia/tritonserver:XX.YY-py3-clientsdk
    ```

1. Create a directory for the test image and put it there. 
**IMPORTANT**: file must have the same name as the input (`DALI_INPUT_0` in our case)
    
    ```
    mkdir test_image
    cp <path to test image> test_image/DALI_INPUT_0
    ```
    
1. Remember, the `test_image` directory must contain **only** the test image
1. Run `perf_analyer`:

    ```
    perf_analyzer -m dali -b 64 --input-data test_image --shape DALI_INPUT_0:`stat --printf="%s" test_image/DALI_INPUT_0`
    ```
    

## Remember

As always with DALI Backend, remember that `dali.fn.external_source`'s `name` parameter must match
with the input name provided in the `config.pbtxt` file.

