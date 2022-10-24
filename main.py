from models.pyramid import Pyramid
from mnist import MNIST

pipeline = MNIST()
model = Pyramid()
pipeline.set(model)

pipeline.test()

# Train the model
# pipeline.train()

# Evaluate the model
# pipeline.eval()

# # Save the model
# pipeline.save_onnx("pyramid.onnx")
