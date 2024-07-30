import torch
import torch.nn as nn
# import torchvision
# import torchvision.transforms as transforms
import torch.nn.functional as F
from networks.cnn import CNN
import numpy as np

PATH_TO_SAVE = './checkpoints/detection/cnn.mdl'


import data_utils.imageDataLoader

train_data, train_labels, test_data, test_labels = data_utils.imageDataLoader.get_data()

train_data, train_labels = torch.Tensor(train_data), torch.Tensor(train_labels).type(torch.LongTensor)
test_data, test_labels = torch.Tensor(test_data), torch.Tensor(test_labels).type(torch.LongTensor)
train_data = train_data.unsqueeze(1)
test_data = test_data.unsqueeze(1)

n_categories = data_utils.imageDataLoader.categories()

print(train_data.shape, train_labels.shape, test_data.shape, test_labels.shape)

# img = torch.Tensor(train_data[0])
# img = img.unsqueeze(0)
# print(img.shape)
# conv1 = nn.Conv2d(1, 6, 5)
# pool2 = nn.MaxPool2d(2, 2)
# conv2 = nn.Conv2d(6, 16, 4)
# pool3 = nn.MaxPool2d(3, 3)

# img = pool2(conv1(img))
# img = pool3(conv2(img))
# print(img.shape)

cnn = CNN()

print("load from saved model [y/n]?")
ans = input()
if ans == 'y':                                      # if saved model is to be used
    cnn.load_state_dict(torch.load(PATH_TO_SAVE))
    cnn.eval()

criterion = nn.CrossEntropyLoss()
learning_rate = 0.007
optimizer = torch.optim.SGD(cnn.parameters(), lr=learning_rate)

num_epochs = 50
batch_size = 10

print_step = 5
running_loss_plot = []

for epoch in range(num_epochs):
    running_loss = 0.0
    i = 0
    for batch_start in range(0, train_data.shape[0], batch_size):
        # print("hello there", i)
        batch_images = train_data[batch_start : batch_start + batch_size]
        batch_labels = train_labels[batch_start : batch_start + batch_size]
        optimizer.zero_grad()
        outputs = cnn(batch_images)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if (i+1) % print_step == 0:
            print(f"EPOCH: {epoch + 1}, step: {i+1}, loss: {(running_loss / print_step):.4f}, test_accuracy: {cnn.get_accuracy(test_data, test_labels):.4f}%")
            running_loss_plot.append(running_loss / print_step)
            running_loss = 0.0
        i += 1

    perm = np.random.permutation(train_data.shape[0])
    train_data = train_data[perm]
    train_labels = train_labels[perm]

import matplotlib.pyplot as plt

plt.plot(running_loss_plot)
plt.title('train loss')
plt.xlabel('iterations')
plt.ylabel('loss')
plt.savefig('./train_loss_cnn.jpg')
plt.show()

## chose if you want to save the model
print("Do you want to save this model [y/n]?")
ans = input()
if ans == 'y':
    torch.save(cnn.state_dict(), PATH_TO_SAVE)






