#!/bin/bash -ex

pushd model_repository

mkdir -p dali/1
mkdir -p ensemble_dali_resnet50/1
mkdir -p resnet50_trt/1

python onnx_exporter.py --save model.onnx
trtexec --onnx=model.onnx --saveEngine=./resnet50_trt/1/model.plan --explicitBatch --minShapes=input:1x3x224x224 --optShapes=input:1x3x224x224 --maxShapes=input:256x3x224x224 --fp16
python serialize_dali_pipeline.py --save ./dali/1/model.dali

echo "Resnet50 model ready."

popd