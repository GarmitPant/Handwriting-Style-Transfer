import torch
import torch.nn as nn

import matplotlib.pyplot as plt

PATH_TO_SAVE = './checkpoints/detection/rnn.mdl'

class DeepIO(nn.Module):
    def __init__(self):
        super(DeepIO, self).__init__()
        self.rnn = nn.LSTM(input_size=3, hidden_size=256,
                           num_layers=2, bidirectional=True)
        self.drop_out = nn.Dropout(0.25)
        self.fc1 = nn.Linear(512, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc_out = nn.Linear(256, 196)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        args:
        x:  a list of inputs of diemension [BxTx6]
        """
        outputs = []
        # iterate in the batch through all sequences
        for xx in x:
            s, n = xx.shape
            out, hiden = self.rnn(xx.unsqueeze(1))
            out = out.view(s, 1, 2, 512)
            out = out[-1, :, 0]
            outputs.append(out.squeeze())
        outputs = torch.stack(outputs)

        y = self.relu(self.fc1(outputs), inplace=True)
        y = self.bn1(y)
        y = self.drop_out(y)
        y = self.out(y)
        return y

class RNN(nn.Module):
    def __init__(self, input_size, prev_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.prev_size = prev_size
        # self.n_layers = n_layers
        self.i2h = nn.Linear((prev_size+1)*input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear((prev_size+1)*input_size + hidden_size, output_size)
        self.tanH = nn.Tanh()
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_tensor, prev_input, hidden_tensor):
        # hidden_tensor = self.init_hidden()
        combined = torch.cat((input_tensor, prev_input.view(1, -1), hidden_tensor), 1)
        # print("hihi ", combined.shape)
        hidden = self.tanH(self.i2h(combined))
        output = self.i2o(combined)
        # output = self.softmax(output)
        return output, hidden
        # batch_size = x.shape[0]
        # hidden = self.init_hidden(batch_size)
        # out, hidden = self.rnn(x, hidden)
        # out = out[:, -1, :]
        # out = self.fc(out)
        # return out
    
    def init_hidden(self):
        # hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        return torch.zeros(1, self.hidden_size)

import data_utils.dataLoaderdetection

train_data, train_labels, test_data, test_labels = data_utils.dataLoaderdetection.get_data()
# print("hi: ",train_data.shape)

n_categories = data_utils.dataLoaderdetection.categories()

def one_hot_encoder(x):
    enc = torch.zeros(1, n_categories)
    enc[0][x] = 1
    return enc

def category_from_output(output):
    category_idx = torch.argmax(output).item()
    return category_idx

input_size = 3
n_hidden = 256
prev_size = 10

rnn = RNN(input_size, prev_size, n_hidden, n_categories)

# rnn.load_state_dict(torch.load(PATH_TO_SAVE))
# rnn.eval()


# # # one step
# hidden_tensor = rnn.init_hidden()
# print(torch.Tensor(train_data[0][0]).view(1, -1).shape)
# output, hidden = rnn(torch.Tensor(train_data[0][0]).view(1, -1), hidden_tensor)
# print(output.shape, hidden.shape)

criterion = nn.CrossEntropyLoss()
learning_rate = 0.001
optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate)

def train(strokeSet, writerId):
    # print("ID:", writerId)
    hidden = rnn.init_hidden()
    prev_input = torch.zeros(prev_size, strokeSet[0].shape[0])
    # print(prev_input.shape)
    for i in range(strokeSet.shape[0]):
        output, hidden = rnn(strokeSet[i].view(1, -1), prev_input, hidden)
        
        for j in range(prev_size-1):
            prev_input[j] = prev_input[j+1]
        prev_input[prev_size-1] = strokeSet[i].view(1, -1)

    loss = criterion(output, torch.Tensor([writerId]).type(torch.LongTensor))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return output, loss.item()

def predict(strokeSet):
    with torch.no_grad():
        hidden = rnn.init_hidden()
        prev_input = torch.zeros(prev_size, strokeSet[0].shape[0])
        # print(prev_input.shape)
        for i in range(strokeSet.shape[0]):
            output, hidden = rnn(strokeSet[i].view(1, -1), prev_input, hidden)
            
            for j in range(prev_size-1):
                prev_input[j] = prev_input[j+1]
            prev_input[prev_size-1] = strokeSet[i].view(1, -1)

        return category_from_output(output)

num_epochs = 100
totalLoss = 0.0
plot_steps, print_steps = 50, 150
losses = []
for epoch in range(num_epochs):
    # print(train_data.shape[0])
    for i in range(train_data.shape[0]):
        # print(train_labels[i])
        output, loss = train(torch.Tensor(train_data[i]), train_labels[i])
        totalLoss += loss

        if (i+1) % plot_steps == 0:
            losses.append(totalLoss / plot_steps)
            totalLoss = 0.0
        
        if (i+1) % print_steps == 0:
            acc = 0
            for strokeSet, actual in zip(test_data, test_labels):
                acc += (1 if predict(torch.Tensor(strokeSet)) == actual else 0)
            acc /= test_data.shape[0]


            guess = category_from_output(output)
            correct = "CORRECT" if guess == train_labels[i] else f"WRONG ({guess, train_labels[i]})"
            print(f"EPOCH: {epoch}, i: {i}, loss: {loss:.4f}, acc: {(acc * 100.0):.4f}")
            # print(f"{i+1} {(i+1)/n_iters*100} {loss:.4f} {line} / {guess} {correct}")

plt.figure()
plt.plot(losses)
plt.show()

print("Do you want to save this model [y/n]?")
ans = input()
if ans == 'y':
    torch.save(rnn.state_dict(), PATH_TO_SAVE)