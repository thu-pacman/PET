import torch
import torchvision.models as models

dummy_input = torch.randn(10, 3, 224, 224)

#resnet18 = models.resnet18(pretrained=True)
#torch.onnx.export(resnet18, dummy_input, "%s.onnx" % ("resnet18"), verbose=False)
#
#alexnet = models.alexnet(pretrained=True)
#torch.onnx.export(alexnet, dummy_input, "%s.onnx" % ("alexnet"), verbose=False)
#
#squeezenet = models.squeezenet1_0(pretrained=True)
#torch.onnx.export(squeezenet, dummy_input, "%s.onnx" % ("squeezenet"), verbose=False)
#
vgg16 = models.vgg16(pretrained=True)
torch.onnx.export(vgg16, dummy_input, "%s.onnx" % ("vgg16"), verbose=False)
#
#densenet = models.densenet161(pretrained=True)
#torch.onnx.export(densenet, dummy_input, "%s.onnx" % ("densenet"), verbose=False)
#
#inception = models.inception_v3(pretrained=True)
#torch.onnx.export(inception, dummy_input, "%s.onnx" % ("inception"), verbose=False)

# googlenet = models.googlenet(pretrained=True)
# torch.onnx.export(googlenet, dummy_input, "%s.onnx" % ("googlenet"), verbose=False)

# shufflenet = models.shufflenet_v2_x1_0(pretrained=True)
# torch.onnx.export(shufflenet, dummy_input, "%s.onnx" % ("shufflenet"), verbose=False)

# mobilenet = models.mobilenet_v2(pretrained=True)
# torch.onnx.export(mobilenet, dummy_input, "%s.onnx" % ("mobilenet"), verbose=False)

# resnext50_32x4d = models.resnext50_32x4d(pretrained=True)
# torch.onnx.export(resnext50_32x4d, dummy_input, "%s.onnx" % ("resnext50_32x4d"), verbose=False)

# wide_resnet50_2 = models.wide_resnet50_2(pretrained=True)
# torch.onnx.export(wide_resnet50_2, dummy_input, "%s.onnx" % ("wide_resnet50_2"), verbose=False)

# mnasnet = models.mnasnet1_0(pretrained=True)
# torch.onnx.export(mnasnet, dummy_input, "%s.onnx" % ("mnasnet"), verbose=False)

