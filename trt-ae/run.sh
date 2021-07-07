#!/bin/bash

if [ ! -n "$PET_HOME" ]; then  
  echo "PET_HOME is not set"  
  exit
fi

make -j

PET_MODEL=$PET_HOME/models-ae/

# Run TensrRT
echo Resnet18
./runtrt-4d ${PET_MODEL}/resnet18_bs1.onnx 1 3 224
./runtrt-4d ${PET_MODEL}/resnet18_bs16.onnx 16 3 224
echo CSRNet
./runtrt-4d ${PET_MODEL}/csrnet_bs1.onnx 1 512 14
./runtrt-4d ${PET_MODEL}/csrnet_bs16.onnx 16 512 14
echo Inception
./runtrt-4d ${PET_MODEL}/inception_bs1.onnx 1 3 330
./runtrt-4d ${PET_MODEL}/inception_bs16.onnx 16 3 330
echo Bert
./runtrt-3d ${PET_MODEL}/bert_bs1.onnx 1 512 768
./runtrt-3d ${PET_MODEL}/bert_bs16.onnx 16 512 768
