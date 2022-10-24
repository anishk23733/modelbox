import torchvision.datasets as datasets
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.onnx
import torch

class MNIST():
    def __init__(self, batch_size=64, num_workers=4, device='cpu'):
        self.device = device

        mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=None)
        mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=None)

        self.x_train = torch.div(mnist_trainset.data, 255).to(device)
        self.y_train = mnist_trainset.targets.to(device)

        self.x_test = torch.div(mnist_testset.data, 255).to(device)
        self.y_test = mnist_testset.targets.to(device)

    def set(self, model):
        self.model = model
        model.to(self.device)

    def test(self):
        print(self.model(self.x_train[0].flatten()))
    
    def save(model, path):
        torch.save(model.state_dict(), path)

    def save_onnx(model, path, x_train):
        torch.onnx.export(model, x_train[0].flatten(), path)

    def load(model, path):
        model.load_state_dict(torch.load(path))
        return model

    def train(model, x_train, y_train):
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

    def eval(model, x_test, y_test):
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

    def show_image(x, i):
        plt.imshow(x[i])
        plt.show()