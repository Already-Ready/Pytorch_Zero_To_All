import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import random
from torch.utils.data import Dataset, DataLoader

# device and seed configure
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# for reproducibility
random.seed(777)
torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)

# parameters
learning_rate = 0.001
epochs = 15
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


# nn Linear layer 만들기
linear1 = torch.nn.Linear(784, 256, bias=True)
linear2 = torch.nn.Linear(256, 256, bias=True)
linear3 = torch.nn.Linear(256, 10, bias=True)
relu = torch.nn.ReLU()

# Linear layer의 weight를 normalization시켜주기
torch.nn.init.normal_(linear1.weight)
torch.nn.init.normal_(linear2.weight)
torch.nn.init.normal_(linear3.weight)

# 모델 정의
model = torch.nn.Sequential(linear1, relu, linear2, relu, linear3).to(device)
###>>> 왜 linear3다음에는 relu를 하지않는가
###>>> 우리가 사용할 criterion은 CrossEntropyLoss인데 여기에는 마지막에 softmax activation이 포함되어 있기 때문이다.

# loss / optimizer 정의
criterion = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

total_batch = len(data_loader)
for epoch in range(epochs):
    avg_loss = 0
    for X, Y in data_loader:
        X = X.view(-1, 28*28).to(device)
        Y = Y.to(device)

        hypothesis = model(X)
        loss = criterion(hypothesis, Y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_loss += loss / total_batch

    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_loss))

print('Learning finished')


with torch.no_grad(): # --> 여기에서는(test에서는) gradient를 계산하지 않고 진행한다는 뜻이다.
    X_test = mnist_test.test_data.view(-1, 28*28).float().to(device)
    Y_test = mnist_test.test_labels.to(device)

    prediction = model(X_test)
    correct_prediction = torch.argmax(prediction, dim=1) == Y_test
    accuracy = correct_prediction.float().mean()
    print('Accuracy:', accuracy.item())

    r = random.randint(0, len(mnist_test) - 1)
    X_single_data = mnist_test.test_data[r:r + 1].view(-1, 28 * 28).float().to(device)
    Y_single_data = mnist_test.test_labels[r:r + 1].to(device)

    print('Label: ', Y_single_data.item())
    single_prediction = model(X_single_data)
    print('Prediction: ', torch.argmax(single_prediction, 1).item())


















