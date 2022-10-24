import torchvision.datasets as datasets
import torch
from utils import train_model, eval_model, save_model, save_onnx, load_model

mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=None)
mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=None)

x_train = torch.div(mnist_trainset.data, 255)
y_train = mnist_trainset.targets

x_test = torch.div(mnist_testset.data, 255)
y_test = mnist_testset.targets

train = lambda model: train_model(model, x_train, y_train)
eval = lambda model: eval_model(model, x_test, y_test)
save_onnx = lambda model, path: save_onnx(model, path, x_train)

from models.pyramid import Pyramid

model = Pyramid()

train(model)
eval(model)