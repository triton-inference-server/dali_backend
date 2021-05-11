#!/bin/bash -ex

GRPC_ADDR:=${1:-"localhost:8001"}

python ensemble_client.py --batch_size 64 --n_iter 3 --model_name ensemble_dali_inception --img_dir images -u $GRPC_ADDR
