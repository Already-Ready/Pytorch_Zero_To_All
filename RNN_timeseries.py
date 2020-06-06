import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)

xy = np.loadtxt("data/data-02-stock_daily.csv", delimiter=",")
xy = xy[::-1]  # reverse order

# 데이터를 0과 1 사이의 값으로 스케일링해주는 함수
def minmax_scaler(data):
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    return numerator / (denominator + 1e-7)

# 데이터를 X와 Y데이터로 분리해주는 함수
def build_dataset(time_series, seq_length):
    dataX = []
    dataY = []
    for i in range(0, len(time_series) - seq_length):
        _x = time_series[i:i + seq_length, :]
        _y = time_series[i + seq_length, [-1]]  # Next close price
        dataX.append(_x)
        dataY.append(_y)
    return np.array(dataX), np.array(dataY)

#-parameters---
seq_length = 7
data_dim = 5
hidden_dim = 10
output_dim = 1
learning_rate = 0.01
iterations = 500
#---------------

# 70프로 train 30프로 test로 분할
train_size = int(len(xy)*0.7) #70프로만 train데이터로 사용
train_set = xy[0:train_size]
test_set = xy[train_size-seq_length:]

# minmax 조절
train_set = minmax_scaler(train_set)
test_set = minmax_scaler(test_set)

# X,Y데이터 분리
train_X, train_Y = build_dataset(train_set, seq_length)
test_X, test_Y = build_dataset(test_set, seq_length)

train_X_tensor = torch.FloatTensor(train_X)
train_Y_tensor = torch.FloatTensor(train_Y)
test_X_tensor = torch.FloatTensor(test_X)
test_Y_tensor = torch.FloatTensor(test_Y)

# 모델 생성
class Net(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, layers):
        super(Net, self).__init__()
        self.rnn = torch.nn.LSTM(input_dim, hidden_dim, num_layers=layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim, output_dim, bias=True)

    def forward(self, x):
        x, _status = self.rnn(x)
        x = self.fc(x[:,-1])
        return x

net = Net(data_dim, hidden_dim, output_dim, 1)
criterion = torch.nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=learning_rate)

# 학습
for i in range(iterations):

    optimizer.zero_grad()
    outputs = net(train_X_tensor)
    loss = criterion(outputs, train_Y_tensor)
    loss.backward()
    optimizer.step()
    print(i, loss.item())


# test데이터를 통해 예측한값과 실제 데이터를 그래프로 그려서 비교
plt.plot(test_Y)
plt.plot(net(test_X_tensor).data.numpy())
plt.legend(['original', 'prediction'])
plt.show()
