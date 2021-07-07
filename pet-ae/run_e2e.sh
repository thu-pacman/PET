#!/bin/bash

if [ ! -n "$PET_HOME" ]; then  
    echo "PET_HOME is not set"  
    exit
fi    

PET_BUILD=${PET_HOME}/build/
PET_MODEL=${PET_HOME}/models-ae/

echo resnet
time ${PET_BUILD}/onnx ${PET_MODEL}/resnet18_bs1.onnx | grep -i "best perf"

time ${PET_BUILD}/onnx ${PET_MODEL}/resnet18_bs16.onnx | grep -i "best perf"

echo CSRNet
time ${PET_BUILD}/dilation 1 1  | grep -i "best perf"

time ${PET_BUILD}/dilation 1 16 | grep -i "best perf"

echo Bert
time ${PET_BUILD}/onnx ${PET_MODEL}/csrnet_bs1.onnx | grep -i "best perf"

time ${PET_BUILD}/onnx ${PET_MODEL}/csrnet_bs16.onnx | grep -i "best perf"

echo Inception
time ${PET_BUILD}/onnx ${PET_MODEL}/inception_bs1.onnx | grep -i "best perf"

time ${PET_BUILD}/onnx ${PET_MODEL}/inception_bs16.onnx | grep -i "best perf"
