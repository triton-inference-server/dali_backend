#!/bin/bash -ex

echo "Run test"
python ensemble_client.py --batch_size 64 --n_iter 3 --model_name ensemble_dali_inception --img_dir images
