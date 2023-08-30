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

# IMPORTANT: $3 argument is used for the CI

source setup.sh

pushd ../.. || exit 1  # Repo's root directory

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

if [[ -z "$3" ]]; then
    DOCKER_RUN_ARGS="--gpus all -p8000:8000 -p8001:8001 -p8002:8002 --privileged"
  else
    DOCKER_RUN_ARGS=$3
fi

echo "Assuming that model is configured. Check the model_repository: "
ls -R model_repository

popd || exit 1

MODEL_REPO="$(pwd)/docs/examples/efficientnet/model_repository"

docker run -t -d --rm $DOCKER_RUN_ARGS --name effnet_bench_cnt --shm-size=50g --ulimit memlock=-1 --ulimit stack=67108864 -v $MODEL_REPO:/models nvcr.io/nvidia/tritonserver:23.07-py3 tritonserver --model-repository /models --log-verbose 1 --model-control-mode explicit

echo "Waiting for tritonserver to wake up..."
sleep 20
echo "... should be enough."

popd || exit 1

docker run --net host -t -v $(pwd):/bench -w /bench nvcr.io/nvidia/tritonserver:23.07-py3-sdk bash run-benchmarks.sh

docker kill effnet_bench_cnt
