#!/bin/bash -ex

: ${GRPC_ADDR:=${1:-"localhost:8001"}}

python client.py --image images/baboon.jpg -v -u $GRPC_ADDR
