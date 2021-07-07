#!/bin/bash

make
echo "conv"
python3 conv.py
echo ""
echo "dilated conv"
python3 dilated_conv.py
echo ""
echo "group conv"
python3 group_conv.py
echo ""
echo "gemm"
python3 gemm.py
