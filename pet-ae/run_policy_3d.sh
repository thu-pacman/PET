#!/bin/bash

if [ ! -n "$PET_HOME" ]; then  
    echo "PET_HOME is not set"  
    exit
fi    

PET_BUILD=${PET_HOME}/build
PET_MODEL=${PET_HOME}/models-ae
unset PET_DISABLE_NEQ_OPT
unset PET_DISABLE_NO_NEQ_OPT

echo resnet3d
echo "joint"
${PET_BUILD}/onnx r3d_18.bs1.onnx  | grep -i "best perf"


echo "equivalent"
export PET_DISABLE_NO_NEQ_OPT=1
${PET_BUILD}/onnx r3d_18.bs1.onnx  | grep -i "best perf"
unset PET_DISABLE_NO_NEQ_OPT



echo "non-equivalent"
export PET_DISABLE_EQ_OPT=1
${PET_BUILD}/onnx r3d_18.bs1.onnx  | grep -i "best perf"
unset PET_DISABLE_EQ_OPT

echo "no opt"
export PET_DISABLE_NEQ_OPT=1
export PET_DISABLE_NO_NEQ_OPT=1
${PET_BUILD}/onnx r3d_18.bs1.onnx  | grep -i "best perf"
unset PET_DISABLE_NEQ_OPT
unset PET_DISABLE_NO_NEQ_OPT
