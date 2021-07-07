import torch
import torchvision.models as models

resnet18 = models.resnet18(pretrained=True)
inception = models.inception_v3(pretrained=True)

for batch in [1, 4, 16, 64]:
    dummy_input = torch.randn(batch, 3, 224, 224)
    torch.onnx.export(resnet18, dummy_input, "models/resnet18.n{}.onnx".format(batch), verbose=False)

    dummy_input = torch.randn(batch, 3, 330, 330)
    torch.onnx.export(inception, dummy_input, "models/inception_v3.hw330.n{}.onnx".format(batch), verbose=False)
