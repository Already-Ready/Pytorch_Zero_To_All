import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

x_data = [[1, 2],
          [2, 3],
          [3, 1],
          [4, 3],
          [5, 3],
          [6, 2]]

y_data = [[0],
           [0],
           [0],
           [1],
           [1],
           [1]]

x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)

w = torch.zeros((2,1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

optimizer = optim.SGD([w, b], lr=1)

# hypothesis = 1 / (1 + torch.exp(-(x_train.matmul(w)+b)))

epochs = 1000

for epoch in range(epochs + 1):
    hypothesis = torch.sigmoid(x_train.matmul(w)+b)
    loss = F.binary_cross_entropy(hypothesis, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print('Epoch {:4d}/{} loss : {:.6f}'.format(
            epoch, epochs, loss.item()
        ))

predict = torch.sigmoid(x_train.matmul(w)+b)

print(predict)

predict_binary = (predict >= torch.FloatTensor([0.5])).float()
print(predict_binary)

Is_correct = (predict_binary == y_train).float()
print(Is_correct)