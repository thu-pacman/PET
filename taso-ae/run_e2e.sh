#!/bin/bash

source /home/zly/env/pytorch/bin/activate

if [ ! -n "$PET_HOME" ]; then  
  echo "PET_HOME is not set"  
  exit
fi

PET_MODEL=${PET_HOME}/models-ae

echo Inception
echo "batch size = 1"
python3 ./inceptionv3.py 1 | grep "Cost metrics"
echo "batch size = 16"
python3 ./inceptionv3.py 16 | grep "Cost metrics"

echo Bert
echo "batch size = 1"
python3 ./bert.py 1 | grep "Cost metrics"
echo "batch size = 16"
python3 ./bert.py 16 | grep "Cost metrics"

echo Resnet
echo "batch size = 1"
python3 ./onnx_test.py -f ${PET_MODEL}/resnet18_bs1.onnx | grep "Cost metrics"
echo "batch size = 16"
python3 ./onnx_test.py -f ${PET_MODEL}/resnet18_bs16.onnx | grep "Cost metrics" 

deactivate
