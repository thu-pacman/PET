#!/bin/bash

set -e

mkdir -p models
mkdir -p logs
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CUDA_HOME

echo "RUN csrnet"
for n in 1 4 16 64; do
    python3 runtf.py ./models/dilated.n$n.pb $n 512 14 14 False 2>&1 | tee logs/dilated-$n-tf.log
    python3 runtf.py ./models/dilated.n$n.pb $n 512 14 14 True 2>&1 | tee logs/dilated-$n-xla.log
done

echo "RUN resnet18"
for n in 1 4 16 64; do
    python3 runtf.py ./models/resnet18.n$n.pb $n 3 224 224 False 2>&1 | tee logs/resnet18-$n-tf.log
    python3 runtf.py ./models/resnet18.n$n.pb $n 3 224 224 True 2>&1 | tee logs/resnet18-$n-xla.log
done

echo "RUN inception_v3"
for n in 1 4 16 64; do
    python3 runtf.py ./models/inception_v3.hw330.n$n.pb $n 3 330 330 False 2>&1 | tee logs/inception_v3-$n-tf.log
    python3 runtf.py ./models/inception_v3.hw330.n$n.pb $n 3 330 330 True 2>&1 | tee logs/inception_v3-$n-xla.log
done

echo "RUN bert"
for n in 1 4 16 64; do
    python3 runtf-bert.py ./models/bert.n$n.pb $n 512 768 False 2>&1 | tee logs/bert-$n-tf.log
    python3 runtf-bert.py ./models/bert.n$n.pb $n 512 768 True 2>&1 | tee logs/bert-$n-xla.log
done
