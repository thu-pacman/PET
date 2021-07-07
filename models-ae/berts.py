import torch
import torch
import torch.nn as nn
import torch.nn.functional as F

batch = 16

layer = 12
seqlen = 512
embed = 768
head = 12
hidden = 64


class Layer(nn.Module):
    def __init__(self):
        super(Layer, self).__init__()
        self.q = nn.Linear(embed, head * hidden, False)
        self.k = nn.Linear(embed, head * hidden, False)
        self.v = nn.Linear(embed, head * hidden, False)
        self.ff1 = nn.Linear(embed, embed)
        self.ff2 = nn.Linear(embed, embed)

    def forward(self, x):
        a = self.q(x).reshape(
            (batch, seqlen, head, hidden)).permute(0, 2, 1, 3)
        assert(list(a.size()) == [batch, head, seqlen, hidden])
        b = self.k(x).reshape(
            (batch, seqlen, head, hidden)).permute(0, 2, 3, 1)
        assert(list(b.size()) == [batch, head, hidden, seqlen])
        c = F.softmax(torch.matmul(a, b), dim=3)
        assert(list(c.size()) == [batch, head, seqlen, seqlen])
        d = self.v(x).reshape(
            (batch, seqlen, head, hidden)).permute(0, 2, 1, 3)
        assert(list(d.size()) == [batch, head, seqlen, hidden])
        e = torch.matmul(c, d).permute(
            0, 2, 1, 3).reshape((batch, seqlen, embed))
        f = F.relu(self.ff1(e))
        g = F.relu(self.ff2(f))
        return x + g

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layers = nn.Sequential(*[Layer() for _ in range(layer)])

    def forward(self, x):
        return self.layers(x)

def Bert(bs):
    global batch
    batch = bs
    return Net()
