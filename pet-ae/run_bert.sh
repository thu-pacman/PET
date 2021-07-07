#!/bin/bash

if [ ! -n "$PET_HOME" ]; then  
    echo "PET_HOME is not set"  
    exit
fi    

PET_BUILD=${PET_HOME}/build/
PET_MODEL=${PET_HOME}/models-ae/

echo Bert
time ${PET_BUILD}/onnx ${PET_MODEL}/bert_bs1.onnx | grep -i "best perf"

time ${PET_BUILD}/onnx ${PET_MODEL}/bert_bs16.onnx | grep -i "best perf"
