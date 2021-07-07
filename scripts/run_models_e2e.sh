export CUDA_VISIBLE_DEVICES=1

# Resnet
./onnx /home/zly/Work/ml-opt/benchmark/models/scripts/resnet18.1.onnx
mv res.cu resnet18-1.cu
nvcc -O2 ./resnet18-1.cu -lcublas -lcudnn -lcurand -o resnet18-1
./resnet18-1

./onnx /home/zly/Work/ml-opt/benchmark/models/scripts/resnet18.16.onnx
mv res.cu resnet18-16.cu
nvcc -O2 ./resnet18-16.cu -lcublas -lcudnn -lcurand -o resnet18-16
./resnet18-16

# CSRNet
./dilation 1 1 
mv res.cu dilation-1.cu
nvcc -O2 ./dilation-1.cu -lcublas -lcudnn -lcurand -o dilation-1
./dilation-1

./dilation 1 16
mv res.cu dilation-16.cu
nvcc -O2 ./dilation-16.cu -lcublas -lcudnn -lcurand -o dilation-16
./dilation-16

# BERT
./onnx /home/zly/Work/ml-opt/benchmark/models/scripts/mybert-new.onnx
mv res.cu bert-1.cu
nvcc -O2 ./bert-1.cu -lcublas -lcudnn -lcurand -o bert-1
./bert-1