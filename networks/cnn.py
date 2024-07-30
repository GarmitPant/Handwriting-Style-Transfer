import torch
import torch.nn as nn
# import torchvision
# import torchvision.transforms as transforms
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 4)
        self.pool3 = nn.MaxPool2d(3, 3)
        # self.conv3 = nn.Conv2d(10, 10, 11)
        self.fc1 = nn.Linear(16 * 15 * 65, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = self.pool2(F.relu(self.conv1(x)))
        x = self.pool3(F.relu(self.conv2(x)))
        # x = self.conv3(x)
        x = torch.flatten(x, 1) # x = x.reshape(nxt.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

    def prediction(self, X):
        with torch.no_grad():
            outputs = self(X)
            _, predictions = torch.max(outputs, 1)
            return predictions

    def get_accuracy(self, X, labels):
        predictions = self.prediction(X)
        correct = 0
        for label, pred in zip(labels, predictions):
            if label == pred:
                correct += 1
        accuracy = (correct / len(labels)) * 100.0
        return accuracy
    
    def get_probability_vector(self, X):
        with torch.no_grad():
            outputs = self(X)
            return F.softmax(outputs)
