#!/usr/bin/env bash

dir=$(realpath $(dirname $0)/..)
docker run --gpus device=1 --rm -v $dir:$dir \
    --user $(id -u):$(id -g) -w $dir/tests \
    -e PYTHONPATH=$dir -t \
    pytorch-shan coverage run --source=../ptxl --omit ../ptxl/gan.py,../ptxl/init.py,../ptxl/lr_scheduler.py -m pytest .

docker run --gpus device=1 --rm -v $dir:$dir \
    --user $(id -u):$(id -g) -w $dir/tests \
    -e PYTHONPATH=$dir -t \
    pytorch-shan coverage report -m 
