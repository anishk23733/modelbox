from torch import nn

class ConvNet(nn.Module):
    def __init__(self, dim=784, dropout=0.1):
        super().__init__()
        self.net = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2)

    def forward(self, x):
        return self.net(x)
