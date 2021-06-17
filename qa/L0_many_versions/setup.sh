#!/bin/bash -ex

pushd model_repository

for policy in all latest specific
do
  for version in 1 2 3
  do
    mkdir -p dali_$policy/$version
    python addition_pipeline.py "dali_$policy/$version/model.dali" $(( version * 10 ))
  done
done

echo "Many-versions model ready."

popd
