from models.pyramid import Pyramid
from mnist import MNIST
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

pipeline = MNIST(device=device)
model = Pyramid()
pipeline.set(model)

# pipeline.test()

# Train the model
pipeline.train(epochs=2)

# Evaluate the model
pipeline.eval()
