import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

# 시드 고정
torch.manual_seed(1)

# 데이터
xy = np.loadtxt('data/data-04-zoo.csv', delimiter=',', dtype=np.float32)

x_train = torch.FloatTensor(xy[:, 0:-1])
y_train = torch.LongTensor(xy[:, [-1]]).squeeze()


# 모델 정의
class SoftmaxClassifierModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(16, 7)

    def forward(self, x):
        return self.linear(x)

model = SoftmaxClassifierModel()

# optimizer 정의
optimizer = optim.SGD(model.parameters(), lr=0.1)

epochs = 1000

# 학습
for epoch in range(epochs + 1):

    predict = model(x_train)
    loss = F.cross_entropy(predict, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print('Epoch {:4d}/{} loss : {:.6f}'.format(
            epoch, epochs, loss.item()
        ))













