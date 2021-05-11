#!/bin/bash -ex

: ${GRPC_ADDR:=${1:-"localhost:8001"}}

python multi_input_client.py --batch_size 256 --n_iter 7 -u $GRPC_ADDR
