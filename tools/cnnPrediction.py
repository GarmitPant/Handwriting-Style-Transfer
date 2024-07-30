import torch
from networks.cnn import CNN
PATH_TO_MODEL = './checkpoints/detection/cnn.mdl'

import data_utils.imageDataLoader
import numpy as np
train_data, train_labels, test_data, test_labels = data_utils.imageDataLoader.get_data()
train_data, train_labels = torch.Tensor(train_data), torch.Tensor(train_labels).type(torch.LongTensor)
test_data, test_labels = torch.Tensor(test_data), torch.Tensor(test_labels).type(torch.LongTensor)
train_data = train_data.unsqueeze(1)
test_data = test_data.unsqueeze(1)

n_categories = data_utils.imageDataLoader.categories()
cnn = CNN()

cnn.load_state_dict(torch.load(PATH_TO_MODEL))
cnn.eval()



# with torch.no_grad():
#     print("Accuracy over test-data:", cnn.get_accuracy(test_data, test_labels), "%")


### single image_prediction
def get_probability_single(x): # x is of size (100, 400)
    X = x.unsqueeze(0)
    # X = np.reshape(x, (1,100,400))
    # X = torch.Tensor(X)

    return cnn.get_probability_vector(X)

## for first image in test_data
# print(get_probability_single(test_data[0]), test_labels[0])

## for test_image.png
# import cv2
# img = torch.Tensor(cv2.imread('test_image.png', 0))
# img = img.unsqueeze(0)
# print(get_probability_single(img), cnn.prediction(img.unsqueeze(0)))
