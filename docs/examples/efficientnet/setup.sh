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

set -e

mkdir -p model_repository/efficientnet_ensemble_gpu/1
mkdir -p model_repository/efficientnet_ensemble_cpu/1

if [[ ! -d DeepLearningExamples ]]; then  # assume that DLE has been properly cloned and patched before
  git clone https://github.com/NVIDIA/DeepLearningExamples.git
  pushd DeepLearningExamples || exit 1
  git checkout 36041957
  git am ../0001-Update-requirements-and-add-Dockerfile.bench.patch
  popd || exit 1
fi

pushd DeepLearningExamples/PyTorch/Classification/ConvNets || exit 1
cp ../../../../deploy_on_triton.py .
popd || exit 1

pushd DeepLearningExamples || exit 1

RANDOM_STRING=$(tr -dc A-Za-z0-9 </dev/urandom | head -c 13 ; echo '')
echo "Random container name: $RANDOM_STRING"

docker build -t convnets -f Dockerfile.bench .
docker run --shm-size 8g --gpus all --name $RANDOM_STRING convnets python deploy_on_triton.py --model-name efficientnet-b0 --model-repository /model_repository --batch-size "$1"

popd || exit 1

docker cp $RANDOM_STRING:/model_repository ./
docker rm $RANDOM_STRING

# Set max_batch_size in the remaining models
insert_batch_size () {
  cp "$1"/config.pbtxt.in "$1"/config.pbtxt
  sed -i s/INSERT_BATCH_SIZE/"$2"/g "$1"/config.pbtxt
}
insert_batch_size model_repository/efficientnet_ensemble_gpu "$1"
insert_batch_size model_repository/efficientnet_ensemble_cpu "$1"
insert_batch_size model_repository/preprocessing_gpu "$1"
insert_batch_size model_repository/preprocessing_cpu "$1"
