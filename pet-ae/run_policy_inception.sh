#!/bin/bash

if [ ! -n "$PET_HOME" ]; then  
    echo "PET_HOME is not set"  
    exit
fi    

PET_BUILD=${PET_HOME}/build
PET_MODEL=${PET_HOME}/models-ae
unset PET_DISABLE_EQ_OPT
unset PET_DISABLE_NO_NEQ_OPT

echo "=============Inception"
echo joint
${PET_BUILD}/onnx ${PET_MODEL}/inception_bs1.onnx | grep -i "best perf"



export PET_DISABLE_EQ_OPT=1

echo non-equivalent
${PET_BUILD}/onnx ${PET_MODEL}/inception_bs1.onnx | grep -i "best perf"

unset PET_DISABLE_EQ_OPT



export PET_DISABLE_EQ_OPT=1
export PET_DISABLE_NO_NEQ_OPT=1

echo non-opt
${PET_BUILD}/onnx ${PET_MODEL}/inception_bs1.onnx | grep -i "best perf"

unset PET_DISABLE_EQ_OPT
unset PET_DISABLE_NO_NEQ_OPT


export PET_DISABLE_NO_NEQ_OPT=1

echo equivalent
${PET_BUILD}/onnx ${PET_MODEL}/inception_bs1.onnx | grep -i "best perf"

unset PET_DISABLE_NO_NEQ_OPT


