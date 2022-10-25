from torch import nn

class CNN(nn.Module):
    def __init__(self, dim=784, dropout=0.1, num_channels=4):
        super().__init__()

        """
        -> Defining a 2D convolution layer.
        We start with a 28x28 image.
        With padding, we have a 30x30 image.
        We apply a 3x3 convolution with `num_channels` channels.
        We have a 28x28 image with `num_channels` channels.
        We apply max pooling with a 2x2 kernel.
        We have a 14x14 image with `num_channels` channels.

        -> Defining another 2D convolution layer
        We have a 14x14 image with `num_channels` channels.
        With padding, we have a 16x16 image.
        We apply a 3x3 convolution with 8 channels.
        We have a 14x14 image with `num_channels` channels.
        We apply max pooling with a 2x2 kernel.
        We have a 7x7 image with `num_channels` channels.
        """
        self.cnn_layers = nn.Sequential(
            # Learned parameters
            nn.Conv2d(1, num_channels, kernel_size=3, stride=1, padding=1),
            # No learned parameters -> transformations
            nn.BatchNorm2d(num_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Learned parameters
            nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1),
            # No learned parameters -> transformations
            nn.BatchNorm2d(num_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.linear_layers = nn.Sequential(
            # Learned parameters
            nn.Linear(num_channels * 7 * 7, 10)
        )

    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x

    def test(self, x):
        return self(self.transform(x))

    def transform(self, x):
        return x.view(-1, 1, 28, 28)