import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

xy = np.loadtxt('data/data-03-diabetes.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]
x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)

class BinaryClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(8, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.linear(x))


model = BinaryClassifier()

optimizer = optim.SGD(model.parameters(), lr=1)

epochs = 100

for epoch in range(epochs + 1):

    hypothesis = model(x_train)

    loss = F.binary_cross_entropy(hypothesis, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        prediction = (hypothesis >= torch.FloatTensor([0.5])).float()
        correct_prediction = (prediction == y_train).float()
        accuracy = correct_prediction.sum().item() / len(correct_prediction)

        print('Epoch : {:4d}/{} loss : {:6f} Accuracy : {:2.2f}'.format(
            epoch, epochs, loss.item(), accuracy*100
        ))