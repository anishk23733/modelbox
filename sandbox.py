import torchvision.datasets as datasets
from torch import nn
import torch
from torch.nn.functional import one_hot
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.onnx

mnist_trainset = datasets.MNIST(root='../data', train=True, download=True, transform=None)
mnist_testset = datasets.MNIST(root='../data', train=False, download=True, transform=None)

x_train = torch.div(mnist_trainset.data, 255)
y_train = mnist_trainset.targets

x_test = torch.div(mnist_testset.data, 255)
y_test = mnist_testset.targets

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


class ConvNet(nn.Module):
    def __init__(self, dim=784, dropout=0.1):
        super().__init__()
        self.net = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2)

    def forward(self, x):
        return self.net(x)

def save_model(model, path):
    torch.save(model.state_dict(), path)

def save_onnx(model, path):
    torch.onnx.export(model, x_train[0].flatten(), path)

def load_model(model, path):
    model.load_state_dict(torch.load(path))
    return model

def train_model(model):
    # create a training loop
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 10
    batch_size = 32

    for epoch in range(epochs):
        total_loss = 0
        for i in tqdm(range(0, len(x_train), batch_size)):
            x = x_train[i:i+batch_size].view(-1, 784)
            y = y_train[i:i+batch_size]

            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print("Epoch: {}, Loss: {}".format(epoch, total_loss))
    
    return model

def eval_model(model):
    model.eval()
    # Evaluate the model on testing data
    with torch.no_grad():
        x = x_test.view(-1, 784)
        y = y_test

        y_pred = model(x)
        _, predicted = torch.max(y_pred.data, 1)
        total = y.size(0)
        correct = (predicted == y).sum().item()
        
        print('Correct:', correct, "Total:", total)
        print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

def show_image(i):
    plt.imshow(x_train[i])
    plt.show()