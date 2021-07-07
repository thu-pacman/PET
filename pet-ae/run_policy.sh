#!/bin/bash

if [ ! -n "$PET_HOME" ]; then  
    echo "PET_HOME is not set"  
    exit
fi    

PET_BUILD=${PET_HOME}/build
PET_MODEL=${PET_HOME}/models-ae

unset PET_DISABLE_EQ_OPT
unset PET_DISABLE_NO_NEQ_OPT
./run_policy_resnet.sh
unset PET_DISABLE_EQ_OPT
unset PET_DISABLE_NO_NEQ_OPT
./run_policy_csrnet.sh
unset PET_DISABLE_EQ_OPT
unset PET_DISABLE_NO_NEQ_OPT
./run_policy_bert.sh
unset PET_DISABLE_EQ_OPT
unset PET_DISABLE_NO_NEQ_OPT
./run_policy_inception.sh
unset PET_DISABLE_EQ_OPT
unset PET_DISABLE_NO_NEQ_OPT

# echo resnet
# time ${PET_BUILD}/onnx ${PET_MODEL}/resnet18_bs1.onnx | grep -i "best perf"
# echo CSRNet
# time ${PET_BUILD}/dilation 1 16  | grep -i "best perf"
# echo Bert
# time ${PET_BUILD}/onnx ${PET_MODEL}/bert_bs1.onnx | grep -i "best perf"
# echo Inception
# time ${PET_BUILD}/onnx ${PET_MODEL}/inception_bs1.onnx | grep -i "best perf"
# 
# 
# 
# 
# 
# 
# export PET_DISABLE_NO_NEQ_OPT=1
# 
# echo resnet
# time ${PET_BUILD}/onnx ${PET_MODEL}/resnet18_bs1.onnx | grep -i "best perf"
# echo CSRNet
# time ${PET_BUILD}/dilation 1 16  | grep -i "best perf"
# echo Bert
# time ${PET_BUILD}/onnx ${PET_MODEL}/bert_bs1.onnx | grep -i "best perf"
# echo Inception
# time ${PET_BUILD}/onnx ${PET_MODEL}/inception_bs1.onnx | grep -i "best perf"
# 
# unset PET_DISABLE_NO_NEQ_OPT
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# export PET_DISABLE_EQ_OPT=1
# 
# echo resnet
# time ${PET_BUILD}/onnx ${PET_MODEL}/resnet18_bs1.onnx | grep -i "best perf"
# echo CSRNet
# time ${PET_BUILD}/dilation 1 16  | grep -i "best perf"
# echo Bert
# time ${PET_BUILD}/onnx ${PET_MODEL}/bert_bs1.onnx | grep -i "best perf"
# echo Inception
# time ${PET_BUILD}/onnx ${PET_MODEL}/inception_bs1.onnx | grep -i "best perf"
# 
# unset PET_DISABLE_EQ_OPT
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# export PET_DISABLE_EQ_OPT=1
# export PET_DISABLE_NO_NEQ_OPT=1
# 
# echo resnet
# time ${PET_BUILD}/onnx ${PET_MODEL}/resnet18_bs1.onnx | grep -i "best perf"
# echo CSRNet
# time ${PET_BUILD}/dilation 1 16  | grep -i "best perf"
# echo Bert
# time ${PET_BUILD}/onnx ${PET_MODEL}/bert_bs1.onnx | grep -i "best perf"
# echo Inception
# time ${PET_BUILD}/onnx ${PET_MODEL}/inception_bs1.onnx | grep -i "best perf"
# 
# unset PET_DISABLE_EQ_OPT
# unset PET_DISABLE_NO_NEQ_OPT
