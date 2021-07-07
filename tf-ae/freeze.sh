#!/bin/bash

set -e

mkdir -p models
mkdir -p logs

echo "EXPORT bert"
python3 export-bert.py

echo "EXPORT resnet-18 and inception-v3"
python3 export-torchvision.py

echo "EXPORT dilated"
python3 export-dilated.py

echo "Convert to PB files"
for n in 1 4 16 64; do
    python3 onnx2pb.py models/bert.n$n.onnx models/bert.n$n.pb
    python3 onnx2pb.py models/inception_v3.hw330.n$n.onnx models/inception_v3.hw330.n$n.pb
    python3 onnx2pb.py models/resnet18.n$n.onnx models/resnet18.n$n.pb 
    python3 onnx2pb.py models/dilated.n$n.onnx models/dilated.n$n.pb
done
