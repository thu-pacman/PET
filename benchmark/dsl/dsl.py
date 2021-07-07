class ConvConfig:
    def __init__(self, nchw, frs, stride=1, dilation=1, group=1):
        if len(nchw) != 4:
            raise Exception("nchw must have 4 dimension")
        if len(frs) != 3:
            raise Exception("frs must have 3 dimension")
        if not (type(stride) is int or (type(stride) is tuple and len(stride) == 2)):
            raise Exception("stride must be int or tuple with length 2")
        if not (type(dilation) is int or (type(dilation) is tuple and len(dilation) == 2)):
            raise Exception("dilation must be int or tuple with length 2")
        if not (type(group) is int):
            raise Exception("group must be int")
        self.n, self.c, self.h, self.w = nchw
        self.f, self.r, self.s = frs
        self.sh, self.sw = stride
        self.dh, self.dw = dilation
        self.group = group

class Dimension:
    def __init__(self, name, size, invalid=[]):
        self.name, self.size, self.invalid = name, size, invalid

class Tensor:
    def __init__(self, dims, base, trans):
        self.dims, self.base, self.trans = dims, base, trans

class DSLPart:
    def __init__(self):
        pass

class DSL:
    def __init__(self):
        pass
