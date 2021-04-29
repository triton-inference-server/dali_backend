#!/bin/bash -ex

pushd model_repository

mkdir -p dali_identity/1
python identity_pipeline.py dali_identity/1/model.dali
echo "Identity model ready."

popd
