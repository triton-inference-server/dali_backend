# GPU-accelerated video processing in Triton using DALI

This example presents, how to approach GPU-accelerated video processing in Triton using DALI.

### First steps

1. If you do not know yet what DALI is, please refer to [DALI's getting started guide](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/examples/getting_started.html#Overview)
1. If you are not yet familiar with the idea of using DALI within Triton, please refer to the
[How to use?](https://github.com/triton-inference-server/dali_backend#how-to-use) guide and
[Identity example](https://github.com/triton-inference-server/dali_backend/tree/main/docs/examples/identity).
These two guides explain the concept and are good starting points.

## GPU-accelerated video processing

While in most of data processing libraries (image or video, like PIL, OpenCV etc...), data decoding
is transparent for user, in DALI we do put emphasis in this step. Actually, whenever `cv2.imread()`
function is used, or `PIL.Image.open()`, the decoding happens underneath. In DALI, user explicitly
calls decoding step, thus gaining a opporunity to use accelerate this operation using GPU.

It's no different in video case, although currently we do face a constraint that every sample passed
to a `fn.experimental.decoders.video()` operator must be a whole video file - including the header.
The following is the canonical way to read the video file into byte-buffer, which can be further
passed to a DALI pipeline:

    decoded_video = np.fromfile(video_file_path, dtype=np.uin8)

## The example

### `dali.py`

This example presents a simple video processing pipeline. Its point is to get the encoded video,
decode it, remove [^1] a distortion and return the result. To see the DALI pipeline definition,
please refer to `dali.py` file. You can also notice, that we used two greatly convenient features:
[autoserialization](https://github.com/triton-inference-server/dali_backend#autoserialization)
and [autoconfig](https://github.com/triton-inference-server/dali_backend#configuration-auto-complete).
Explaining these is beyond the scope of this tutorial, please refer to corresponding documentation.
    
### `remap.npz`

In general, removing the distortion in an image consists of a few steps:
1. Determine the camera coefficients,
2. Determine the distortion coefficients (these will be specific to a given physical camera),
3. Calculate the Remap coefficients. This step uses the distortion coefficients to calculate the
Remap parameters. To find out more, please refer to [`fn.remap` documentation](https://docs.nvidia.com/deeplearning/dali/main-user-guide/docs/operations/nvidia.dali.fn.experimental.remap.html)
4. Perform the Remap operation (effectively correcting the distortion).

`remap.npz` file contains already calculated Remap coefficients. That means, we're skipping steps 1, 2 and 3.

### `client.py`

Lastly, our example needs a Triton client. Our client loads the video from local memory (the path
to the videos can be configured with `--videos` argument), sends them to the `tritonserver` instance
and receives the result. Please note, that the script assumes, that the user has the
[DALI_extra repository](https://github.com/NVIDIA/DALI_extra) cloned and the path to this repository
in the local storage is specified in `DALI_EXTRA_PATH` environment variable. If otherwise, please
specify the path to video data manually.

### Running the example

Running the example requires two steps:
#### Step 1: Run the `tritonserver` instance (we'll do it on `localhost`):
    
    MODEL_REPO="<path to dali_backend repository>/docs/examples/video_decode_remap/model_repository" && \
    docker run -it --rm --shm-size=1g --ulimit memlock=-1 --gpus all --ulimit stack=67108864 -p8000:8000 -p8001:8001 -p8002:8002 -v $MODEL_REPO:/models tritonserver:22.12-py3 tritonserver --model-repository /models

#### Step 2: Run the client
Please note that in addition to the command below, the user has to ensure the test data is visible
inside the docker container. Simplest way to do this would be to copy it to the `$CLIENT_PATH/data`.

    CLIENT_PATH="<path to dali_backend repository>/docs/examples/video_decode_remap" && \
    docker run -it -v $CLIENT_PATH:/client tritonserver:22.12-py3-sdk python /client/client.py


## Remember

As always with DALI Backend, remember that `dali.fn.external_source`'s `name` parameter must match
with the input name provided in the `config.pbtxt` file. Additionally, when using Autoconfig, 
the last operator in the pipeline shall have the `name` parameter set and match the name of the output
in the client.

[^1]: Surely you've noticed, that we do not remove the distortion here, but add it. Yes, that's intentional.
It's because the original samples are not distorted, therefore "removing" the barrel distortion on
undistorted images results in pincushion distortion.