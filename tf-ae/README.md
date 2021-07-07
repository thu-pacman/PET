# Install Dependency

1. Install pre-compiled libraries
    ```bash
    pip install tensorflow==1.15.0 tensorflow-gpu==1.15.0 torch torchvision onnx==1.7.0
    ```

2. Install onnx-tf from source (the pre-compiled version depends on TF2)

    ```bash
    git clone https://github.com/onnx/onnx-tensorflow.git
    cd onnx-tensorflow
    git checkout v1.7.0-tf-1.15
    pip install .
    cd ..
    ```

3. Prepare cuda 10.0 and cudnn 7.6

    ```bash
    spack load cuda@10.0.130%gcc@9.2.0
    spack load cudnn@7.6.5.32-10.0-linux-x64%gcc@9.2.0
    ```

4. Put a soft link of ptxas to "./bin/ptxas" as needed by tensorflow

    ```bash
    mkdir bin
    ln -s `which ptxas` bin/ptxas
    ```

# Prepare Models

```bash
chmod +x freeze.sh
./freeze.sh
```

This script uses `torch.onnx.export` to export the bert, inception_v3, and resnet18 with different batch sizes to ONNX format, and use `onnx-tf` to convert the ONNX model to `*.pb` format. Both the ONNX format models and pb format models will be saved in `./models/`

# Run Models

```bash
chmod +x run.sh
./run.sh
```

This script will run the bert, inception_v3, and resnet18 models with XLA turned on and off respectively and record the averaged execution time of 100 runs. The logs will be saved in ./logs/.

