#!/bin/bash -ex

pushd model_repository

mkdir -p dali_multi_input/1
python multi_input_pipeline.py dali_multi_input/1/model.dali
echo "Multi-input model ready."

popd
