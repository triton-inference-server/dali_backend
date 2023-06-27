# ResNet101 PyTorch segmentation example

The example presents an inference scenario using DALI and ResNet101.

ResNet101 is a segmentation model. Together with DALI on board they form the following scenario:

1. **Preprocessing** - DALI-based typical ResNet preprocessing. Instead of images the input data is a video.
   Includes GPU decoding (using NVDEC), resize and normalization.
2. **Inference** - the model returns the probabilities of a given class in every pixel.
3. **Postprocessing** - DALI takes the original image and the probabilities and extracts a particular class.

Every step mentioned above is executed by the Triton server. Triton client is used only for reading the test
data from disk and handling the result.


## Setting up the model

To set up the model, please run `install_dependencies.sh` inside your Triton docker container. This script installs PyTorch, which is a part of this example.


## Running the example

To run the example please follow the list:
1. Clone the repository:
```bash
$ git clone https://github.com/triton-inference-server/dali_backend.git
```
2. Download the latest Triton release:
```bash
$ docker pull nvcr.io/nvidia/tritonserver:23.05-py3
```
3. Run the `tritonserver` container (remember to set the `MODEL_PATH` properly):
```bash
$ MODEL_REPO="//dali_backend/docs/examples/segmentation" && docker run -it --rm --gpus all --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -p8000:8000 -p8001:8001 -p8002:8002 --privileged -v $MODEL_REPO:/segmentation nvcr.io/nvidia/tritonserver:23.05-py3 bash
```
4. Inside the container, install required dependencies
```bash
$ cd /segmentation
$ bash install_dependencies.sh
```
5. Run the `tritonserver`:
```bash
$ tritonserver --model-repository /segmentation/model_repository
```
6. In a separate window run the client script:
```bash
$ cd //dali_backend/qa/L0_segmentation
$ python client.py
```
