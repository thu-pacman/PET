import torch
import torch.nn as nn
import sys

"""
    input shape [-1, 512, 14, 14]

    weights shape
    [512, 512, 3, 3]
    [512, 512, 3, 3]
    [512, 512, 3, 3]
    [256, 512, 3, 3]
    [128, 256, 3, 3]
    [ 64, 128, 3, 3]
"""


def DilatedNet():
    dilaRate = [2, 2, 2, 2, 2, 2]
    pad = dilaRate

    dilaNet = nn.Sequential(
        nn.Conv2d(512, 512, 3, padding=pad[0],
                  dilation=dilaRate[0], bias=False),
        nn.ReLU(),
        nn.Conv2d(512, 512, 3, padding=pad[1],
                  dilation=dilaRate[1], bias=False),
        nn.ReLU(),
        nn.Conv2d(512, 512, 3, padding=pad[2],
                  dilation=dilaRate[2], bias=False),
        nn.ReLU(),
        nn.Conv2d(512, 256, 3, padding=pad[3],
                  dilation=dilaRate[3], bias=False),
        nn.ReLU(),
        nn.Conv2d(256, 128, 3, padding=pad[4],
                  dilation=dilaRate[4], bias=False),
        nn.ReLU(),
        nn.Conv2d(128, 64, 3, padding=pad[5],
                  dilation=dilaRate[5], bias=False),
        nn.ReLU()
    )
    return dilaNet
"""
    dummy_input = torch.randn(batch, 512, 14, 14)

    torch.onnx.export(dilaNet, dummy_input, "%s_%d.onnx" %
                      ("dilation", bs), verbose=False)


if __name__ == "__main__":
    main()
"""