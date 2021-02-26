# perf_analyzer example with DALI image decoder


## Run the example

### tritonserver

To run the tritonserver, first you need to serialize DALI pipeline and put it inside model repo.

`setup_perf_analyzer_example.sh` is a convenience script, that automatizes setting up.
Provided you have DALI wheel installed in your system, you can just call `sh ./setup_perf_analyzer_example.sh`

When you have your model repo set up, you can run `tritonserver`. Be sure to fill `<path to model repo>` properly:

    docker run -it --rm --shm-size=1g --ulimit memlock=-1 --gpus all --ulimit stack=67108864 -p8000:8000 -p8001:8001 -p8002:8002 --privileged -v <path to model repo>:/models nvcr.io/nvidia/tritonserver:XX.YY-py3 tritonserver --model-repository=/models

### perf_analyzer

1. Pick an image you like the most
1. Run Triton's client docker container:
 
    ```
    docker run -it --net=host nvcr.io/nvidia/tritonserver:XX.YY-py3-clientsdk
    ```

1. Create a directory for the test image and put it there. 
**IMPORTANT**: file must have the same name as the input (`DALI_INPUT_0` in our case)
    
    ```
    mkdir test_image
    touch test_image/DALI_INPUT_0  # it's not really "touch" - it's an image
    ```
    
1. Remember, the `test_image` directory must contain **only** the test image
1. Run `perf_analyer`:

    ```
    perf_analyzer -m dali -b 64 --input-data test_image --shape DALI_INPUT_0:`stat --printf="%s" test_image/DALI_INPUT_0`
    ```
    

## Remember

As always in DALI Backend case, remember that `dali.fn.external_source`'s `name` parameter must match
with input name provided in `config.pbtxt` file.

