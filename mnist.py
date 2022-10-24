import torchvision.datasets as datasets
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.onnx
import torch
from torch import nn

class MNIST():
    def __init__(self, device='cpu'):
        self.device = device

        mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=None)
        mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=None)

        self.x_train = torch.div(mnist_trainset.data, 255).to(device)
        self.y_train = mnist_trainset.targets.to(device)

        self.x_test = torch.div(mnist_testset.data, 255).to(device)
        self.y_test = mnist_testset.targets.to(device)

    def set(self, model):
        self.model = model
        self.model.to(self.device)

    def test(self):
        print(self.model(self.x_train[0].flatten()))
    
    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def save_onnx(self, path):
        torch.onnx.export(self.model, self.x_train[0].flatten(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))
        return self.model

    def train(self, epochs=10, batch_size=32):
        # create a training loop
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        for epoch in range(epochs):
            total_loss = 0
            for i in tqdm(range(0, len(self.x_train), batch_size)):
                x = self.x_train[i:i+batch_size].view(-1, 784)
                y = self.y_train[i:i+batch_size]

                y_pred = self.model(x)
                loss = loss_fn(y_pred, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            print("Epoch: {}, Loss: {}".format(epoch, total_loss))
        
        return self.model

    def eval(self):
        self.model.eval()
        # Evaluate the model on testing data
        with torch.no_grad():
            x = self.x_test.view(-1, 784)
            y = self.y_test

            y_pred = self.model(x)
            _, predicted = torch.max(y_pred.data, 1)
            total = y.size(0)
            correct = (predicted == y).sum().item()
            
            print('Correct:', correct, "Total:", total)
            print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

    def show_image(self, i):
        plt.imshow(self.x_train[i])
        plt.show()