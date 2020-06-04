import torch
import torch.optim as optim
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)


sentence = ("if you want to build a ship, don't drum up people together to "
            "collect wood and don't assign them tasks and work, but rather "
            "teach them to long for the endless immensity of the sea.")

char_set = list(set(sentence))
char_dic = {c:i for i, c in enumerate(char_set)}

dic_size = len(char_dic)
hiddend_size = len(char_dic)
sequence_length = 10
learning_rate = 0.1

x_data = []
y_data = []

for i in range(0, len(sentence) - sequence_length):
    x = sentence[i: i+sequence_length]
    y = sentence[i+1: i+sequence_length+1]

    x_data.append([char_dic[c] for c in x])
    y_data.append([char_dic[c] for c in y])

x_one_hot = [np.eye(dic_size)[x] for x in x_data]


X = torch.FloatTensor(x_one_hot).to(device)
Y = torch.LongTensor(y_data).to(device)


# RNN 모델
class Net(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, layers):
        super(Net, self).__init__()
        self.rnn = torch.nn.RNN(input_dim, hidden_dim, num_layers=layers, batch_first=True)
        # num_layers를 설정함으로써 RNN layer를 여러겹 쌓을 수 있다.
        self.fc = torch.nn.Linear(hidden_dim, hidden_dim, bias=True)

    def forward(self, x):
        x, _status = self.rnn(x)
        x = self.fc(x)
        return x

net = Net(dic_size, hiddend_size, 2).to(device)

criterion = torch.nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(net.parameters(), lr=learning_rate)

for i in range(100):
    optimizer.zero_grad()
    outputs = net(X)
    loss = criterion(outputs.view(-1,dic_size), Y.view(-1))
    loss.backward()
    optimizer.step()

    results = outputs.argmax(dim=2)
    predict_str = ''

    for j, result in enumerate(results):
        if j==0:
            predict_str += ''.join([char_set[t] for t in result])
        else:
            predict_str += char_set[result[-1]]

    print(predict_str)