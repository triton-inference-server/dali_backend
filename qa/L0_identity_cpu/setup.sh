#!/bin/bash -ex

pushd model_repository

mkdir -p dali_identity_cpu/1
python identity_pipeline.py dali_identity_cpu/1/model.dali
mkdir -p dali_identity_cpu_dynamic/1
cp dali_identity_cpu/1/model.dali dali_identity_cpu_dynamic/1/model.dali
echo "Identity model ready."

popd
