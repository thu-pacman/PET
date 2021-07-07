import torch
import torchvision.models as models
from berts import Bert
from dilated_onnx import csrnet

# 330 330 inception
# 224 224 other
# 16 1
# resnet inception
import argparse

# model_list = ["resnet18", "inception", "bert-opt", "bert-onnx", "csrnet"]
model_list = ["resnet18", "inception", "bert", "csrnet"]
batchsize_list = [1, 16]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="dump onnx")
    parser.add_argument("--batchsize", type=int, default=1)
    parser.add_argument("--model", type=str, required=True)
    #parser.add_argument("--width", type=int, default=224)
    #parser.add_argument("--height", type=int, default=224)
    #parser.add_argument("--channel", type=int, default=3)
    args = parser.parse_args()
    # print(args)

    assert args.model in model_list, "model must be in {}".format(
        str(model_list))
    assert args.batchsize in batchsize_list, "batchsize must be in {}".format(
        str(batchsize_list))

    if args.model == "inception":
        param = torch.randn(args.batchsize, 3, 330, 330)
        model = models.inception_v3(pretrained=True)
    elif args.model == "bert":
        param = torch.randn(args.batchsize, 512, 768)
        # if args.model == "bert-opt":
        #     model = BertOptNet(batch=args.batchsize)
        # else:
        model = Bert(args.batchsize)
    elif args.model == 'csrnet':
        param = torch.randn(args.batchsize, 512, 14, 14)
        model = csrnet()
    elif args.model == 'resnet18':
        param = torch.randn(args.batchsize, 3, 224, 224)
        model = models.resnet18(pretrained=True)
    else:
        assert(False)

    torch.onnx.export(model, param, "{}_bs{}.onnx".format(
        args.model, args.batchsize), verbose=False)
