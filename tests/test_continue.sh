#!/usr/bin/env bash

dir=$(realpath $(dirname $0)/..)

docker run --gpus device=1 --rm -v $dir:$dir \
    --user $(id -u):$(id -g) \
    -e PYTHONPATH=$dir -w $dir/tests -t \
    pytorch-shan:1.6.0-cuda10.1-cudnn7-runtime ./test_continue.py
