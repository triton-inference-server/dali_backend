#!/bin/bash -ex

source setup.sh

pushd ../.. || exit 1  # Repo's root directory

pushd docs/examples/efficientnet || exit 1

source setup.sh
echo "Model configured. Check the model_repository: "
ls -R model_repository

popd || exit 1

MODEL_REPO="$(pwd)/docs/examples/efficientnet/model_repository"

docker run -t -d --rm --name effnet_bench_cnt --gpus all --shm-size=50g --ulimit memlock=-1 --ulimit stack=67108864 -p8000:8000 -p8001:8001 -p8002:8002 --privileged -v $MODEL_REPO:/models nvcr.io/nvidia/tritonserver:23.06-py3 tritonserver --model-repository /models --log-verbose 1 --model-control-mode explicit

echo "Waiting for tritonserver to wake up..."
sleep 10
echo "... should be enough."

popd || exit 1

docker run --net host -it -v $(pwd):/bench -w /bench nvcr.io/nvidia/tritonserver:23.06-py3-sdk bash run-benchmarks.sh

docker kill effnet_bench_cnt