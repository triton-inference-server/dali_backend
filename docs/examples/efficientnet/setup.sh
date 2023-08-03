pushd examples/public/PyTorch/Classification/ConvNets || exit
wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/efficientnet_b4_pyt_amp/versions/20.12.0/zip -O efficientnet_b4_pyt_amp_20.12.0.zip
unzip -o efficientnet_b4_pyt_amp_20.12.0.zip

docker build -t convnets -f Dockerfile .
docker run -di --shm-size 8g --gpus all --name convnets_container convnets
docker exec convnets_container python deploy_on_triton.py --model-name efficientnet-b4 --checkpoint nvidia_efficientnet-b4_210412.pth --model-repository ./model_repository

popd || exit

docker cp convnets_container:/workspace/model_repository ./
