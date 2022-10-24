from torch import nn
import torch
import torch.onnx
from tqdm import tqdm
import matplotlib.pyplot as plt

def save_model(model, path):
    torch.save(model.state_dict(), path)

def save_onnx(model, path, x_train):
    torch.onnx.export(model, x_train[0].flatten(), path)

def load_model(model, path):
    model.load_state_dict(torch.load(path))
    return model

def train_model(model, x_train, y_train):
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

def eval_model(model, x_test, y_test):
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