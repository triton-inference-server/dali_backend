#!/bin/bash -ex

pushd model_repository
mkdir -p inception_graphdef/1
mkdir -p dali/1
mkdir -p ensemble_dali_inception/1

wget -O /tmp/inception_v3_2016_08_28_frozen.pb.tar.gz \
     https://storage.googleapis.com/download.tensorflow.org/models/inception_v3_2016_08_28_frozen.pb.tar.gz
(cd /tmp && tar xzf inception_v3_2016_08_28_frozen.pb.tar.gz)
mv /tmp/inception_v3_2016_08_28_frozen.pb inception_graphdef/1/model.graphdef

python inception_pipeline.py dali/1/model.dali
popd
