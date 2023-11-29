import torch.nn as nn
import torch.optim as optim

class SimpleNN(nn.Module):
    def __init__(self, input_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x
    
    ##############################
def incremental_train(model, criterion, optimizer, new_X, new_y):
    model.train()
    optimizer.zero_grad()
    outputs = model(new_X)
    loss = criterion(outputs, new_y.view(-1, 1))
    loss.backward()
    optimizer.step()