from torch import nn

class Pyramid(nn.Module):
    def __init__(self, dim=784, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential()
        
        layers = []
        n = dim
        i = 0
        while n > 40:
            self.net.add_module("linear_{}".format(n), nn.Linear(n, n // 2))
            self.net.add_module("relu_{}".format(n), nn.ReLU())
            
            if i % 2 == 1: self.net.add_module("dropout_{}".format(n), nn.Dropout(dropout))

            n = n // 2
            i += 1
        
        self.net.add_module("linear_{}".format(n), nn.Linear(n, 10))
        self.net.add_module("relu_{}".format(n), nn.ReLU())

    def forward(self, x):
        return self.net(x)

    def test(self, x):
        return self(self.transform(x))

    def transform(self, x):
        return x.view(-1, 28 * 28)