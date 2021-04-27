#!/bin/bash -ex

echo "Test run"
python multi_input_client.py --batch_size 256 --n_iter 7
