#!/bin/bash -ex

: ${GRPC_ADDR:=${1:-"localhost:8001"}}

python many_versions_client.py -u $GRPC_ADDR
