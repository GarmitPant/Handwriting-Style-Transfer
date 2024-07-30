import numpy as np

writer_data = np.load('./data/detection/lineStroke.npy', allow_pickle=True)

def get_data():

    TRAIN_PERCENT = 0.7

    train_data = []
    train_labels = []

    test_data = []
    test_labels = []

    for writerId in range(10):
        numWritten = writer_data[writerId].shape[0]
        partition = int(numWritten * TRAIN_PERCENT)
        for i in range(partition):
            train_data.append(writer_data[writerId][i])
            train_labels.append(writerId)
        for i in range(partition, numWritten):
            test_data.append(writer_data[writerId][i])
            test_labels.append(writerId)

    train_data, train_labels = np.array(train_data, dtype=object), np.array(train_labels, dtype=object)
    test_data, test_labels = np.array(test_data, dtype=object), np.array(test_labels, dtype=object)
    # print(len(test_data), len(test_labels))
    perm = np.random.permutation(len(train_data))
    train_data, train_labels = train_data[perm], train_labels[perm]

    perm = np.random.permutation(len(test_data))
    test_data, test_labels = test_data[perm], test_labels[perm]

    return train_data, train_labels, test_data, test_labels

def categories():
    return 10 #writer_data.shape[0]

    # print(train_data[0].shape, train_labels[0])