import numpy as np

writer_data = np.load('./data/detection/lineImages.npy', allow_pickle=True)

def get_data():
    TRAIN_PERCENT = 0.7

    train_data = []
    train_labels = []

    test_data = []
    test_labels = []

    for writerId in range(1, 11):
        numWritten = writer_data[writerId].shape[0]
        # print(numWritten)
        partition = int(numWritten * TRAIN_PERCENT)
        # perm = np.random.permutation(numWritten)
        # writer_data[writerId] = writer_data[writerId][perm]
        for i in range(partition):
            train_data.append(writer_data[writerId][i])
            train_labels.append(writerId-1)
        for i in range(partition, numWritten):
            test_data.append(writer_data[writerId][i])
            test_labels.append(writerId-1)

    train_data, train_labels = np.array(train_data), np.array(train_labels)
    test_data, test_labels = np.array(test_data), np.array(test_labels)
    # print(len(test_data), len(test_labels))
    perm = np.random.permutation(len(train_data))
    train_data, train_labels = train_data[perm], train_labels[perm]

    perm = np.random.permutation(len(test_data))
    test_data, test_labels = test_data[perm], test_labels[perm]

    return train_data, train_labels, test_data, test_labels

def categories():
    return 10 #writer_data.shape[0]

    # print(train_data[0].shape, train_labels[0])

# train_data, _, test_data, _ = get_data()
# print(train_data.shape, test_data.shape)