import torchvision.datasets as dsets
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.nn as nn
import matplotlib.pyplot as plt
import random
import numpy as np
import torch

# device and seed configure
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# for reproducibility
random.seed(777)
torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)

# parameters
training_epochs = 5
batch_size = 100

# dataset
mnist_train = dsets.MNIST(root='MNIST_data/', train=True,
                          transform=transforms.ToTensor(), download=True)
mnist_test = dsets.MNIST(root='MNIST_data/', train=False,
                          transform=transforms.ToTensor(), download=True)


# dataloader
data_loader = DataLoader(dataset=mnist_train, batch_size=batch_size, shuffle=True, drop_last=True)
# drop_last 는 batch_size에 맞춰서 이미지를 나눠서 가지고 오고 마지막에 batch_size에 맞지 않아 남은 이지미들을 어떻게
# 처리할지를 다루는 인자인데, True를 사용하면 나머지 이미지를 사용하지 않겠다는 뜻이다.

# batch_size로 이미지를 가져오게 되면
# (batch size, channel, Height, Width) 로 가져오게 된다.
# (batch size, 1, 28, 28) 이미지를 view를 통해 (batch size, 28*28) 로 바꿔줄 것이다

# define model / optimizer / loss
class MultivariateLinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(784, 10, bias=True).to(device)

    def forward(self, x):
        return self.linear(x)

model = MultivariateLinearRegressionModel()
criterion = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# train
for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = len(data_loader)

    for X, Y in data_loader:
        X = X.view(-1, 28 * 28).to(device) # --> (batch_size, 784) 텐서로 변환
        Y = Y.to(device)

        optimizer.zero_grad()
        hypothesis = model(X)
        cost = criterion(hypothesis, Y)
        cost.backward()
        optimizer.step()

        avg_cost += cost / total_batch

    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

print('Learning finished')

# test and visualization
with torch.no_grad(): # --> 여기에서는(test에서는) gradient를 계산하지 않고 진행한다는 뜻이다.
    X_test = mnist_test.test_data.view(-1, 28*28).float().to(device)
    Y_test = mnist_test.test_labels.to(device)

    prediction = model(X_test)
    correct_prediction = torch.argmax(prediction, dim=1) == Y_test
    accuracy = correct_prediction.float().mean()
    print('Accuracy:', accuracy.item())

    # get one image / prediction and plot
    r = random.randint(0, len(mnist_test) - 1)
    X_single_data = mnist_test.test_data[r:r + 1].view(-1, 28 * 28).float().to(device)
    Y_single_data = mnist_test.test_labels[r:r + 1].to(device)

    print('Label: ', Y_single_data.item())
    single_prediction = model(X_single_data)
    print('Prediction: ', torch.argmax(single_prediction, 1).item())

    plt.imshow(mnist_test.test_data[r:r + 1].view(28, 28), cmap='Greys', interpolation='nearest')
    plt.show()











