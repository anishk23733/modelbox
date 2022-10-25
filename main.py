from models.pyramid import Pyramid
from models.conv import ConvNet
from mnist import MNIST
import torch

device = torch.device(5 if torch.cuda.is_available() else "cpu")

pipeline = MNIST(device=device)
model = ConvNet()
pipeline.set(model)

res = model.test(pipeline.x_train[0])
# print(res)
# print(res.shape)

# Train the model
pipeline.train(epochs=10, batch_size=2048)

# Evaluate the model
pipeline.eval()
