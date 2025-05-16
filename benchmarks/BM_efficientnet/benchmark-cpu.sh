# The MIT License (MIT)
#
# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

# Usage:
# bash benchmark-cpu.sh                             -> Run the benchmark, assuming that it is properly initialized.
# bash benchmark-cpu.sh do_setup [MAX_BATCH_SIZE]   -> Run the benchmark, initializing it beforehand.

source setup.sh

pushd ../.. || exit 1 # Repo's root directory

pushd docs/examples/efficientnet || exit 1

is_number_re='^[0-9]+$'

if [[ "$1" == "do_setup" ]]; then
  if ! [[ $2 =~ $is_number_re ]]; then
    echo "Please provide the max batch size."
  else
    echo "Configuring inference with max batch size: $2"
  fi
  source setup.sh "$2"
fi

echo "Assuming that model is configured. Check the model_repository: "
ls -R model_repository

popd || exit 1

MODEL_REPO="$(pwd)/docs/examples/efficientnet/model_repository"
SERVER_CONTAINER_NAME="efficientnet.server"
CLIENT_CONTAINER_NAME="efficientnet.client"
CPU_IMGE_NAME="tritonserver_torchvision"

pushd benchmarks/BM_efficientnet || exit 1

docker build -t $CPU_IMGE_NAME -f Dockerfile.torch .

popd || exit 1

docker run -dt --rm $DOCKER_RUN_ARGS --name ${SERVER_CONTAINER_NAME} --shm-size=50g --ulimit memlock=-1 --ulimit stack=67108864 --gpus all -p8000:8000 -p8001:8001 -p8002:8002 --privileged -v $MODEL_REPO:/model_repository $CPU_IMGE_NAME tritonserver --model-repository /model_repository --log-verbose 1 --model-control-mode explicit

echo "Waiting for tritonserver to wake up..."
sleep 30
echo "... should be enough."

popd || exit 1

docker run -t --rm --name ${CLIENT_CONTAINER_NAME} --net host -v $(pwd):/bench -w /bench nvcr.io/nvidia/tritonserver:25.04-py3-sdk bash run-benchmarks-cpu.sh $2

docker kill ${SERVER_CONTAINER_NAME}
