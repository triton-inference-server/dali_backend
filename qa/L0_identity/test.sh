#!/bin/bash -ex

echo "Test setup"
pushd model_repository
source setup.sh
popd

echo "Test run"
python identity_client.py
