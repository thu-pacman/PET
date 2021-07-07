## Instruction for running TensorRT experiments

### Requirements

* TensorRT-7.0.0.11
* CUDNN 7
* CUDA 10.2
* python3
    * torch
    * torch-vision
    * ONNX

### How to run

1. Change the CUDA_INSTALL_PATH to your CUDA path (e.g. /path/cuda-10.2)
2. Change the CUDNN_INSTALL_PATH to your CUDNN path (e.g. /path/cudnn-7.6.5)
3. Change the TRT_INSTALL_PATH to your tensorRT path (e.g. /path/TensorRT-7.0.0.11)

```bash
TRT_INSTALL_PATH=/path/TensorRT-7.0.0.11 ./run.sh
# In total, we will run four models with two kinds of batchsize (1 and 16)
```
