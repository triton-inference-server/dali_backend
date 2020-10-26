# The MIT License (MIT)
#
# Copyright (c) 2020 NVIDIA CORPORATION
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

ARG BASE_IMAGE=nvcr.io/nvidia/tritonserver:20.09-py3
FROM ${BASE_IMAGE} as builder

RUN apt-get update && \
    apt-get install -y software-properties-common

RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y  \
              zip              \
              wget             \
              build-essential  \
              autoconf         \
              autogen          \
              unzip            \
              python3.8        \
              python3-pip      \
              libboost-all-dev \
              rapidjson-dev

# pip version in apt packages is ancient - we need to update it
RUN pip3 install -U pip

# CMake
RUN CMAKE_VERSION=3.17 && \
    CMAKE_BUILD=3.17.4 && \
    wget -nv https://cmake.org/files/v${CMAKE_VERSION}/cmake-${CMAKE_BUILD}.tar.gz && \
    tar -xf cmake-${CMAKE_BUILD}.tar.gz && \
    cd cmake-${CMAKE_BUILD} && \
    ./bootstrap --parallel=$(grep ^processor /proc/cpuinfo | wc -l) -- -DCMAKE_USE_OPENSSL=OFF && \
    make -j"$(grep ^processor /proc/cpuinfo | wc -l)" install && \
    rm -rf /cmake-${CMAKE_BUILD}

RUN pip download -d /dali_whl --extra-index-url https://developer.download.nvidia.com/compute/redist/nightly nvidia-dali-nightly-cuda110 && \
    pip install --force-reinstall /dali_whl/nvidia_dali* && mkdir -p /dali && \
    unzip /dali_whl/nvidia_dali* -d /dali

WORKDIR /dali_backend

COPY . .

RUN mkdir build_in_ci && cd build_in_ci && \
    cmake -D CMAKE_BUILD_TYPE=Release -D TRITON_DALI_SKIP_DOWNLOAD=ON .. && make -j"$(grep ^processor /proc/cpuinfo | wc -l)"


FROM ${BASE_IMAGE} as tritonserver_dali

COPY --from=builder /dali/nvidia/dali/.libs /dali/.libs
COPY --from=builder /dali/nvidia/dali/*.so /dali/
COPY --from=builder /dali_backend/build_in_ci/src/libtriton_dali.so /

ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/dali
