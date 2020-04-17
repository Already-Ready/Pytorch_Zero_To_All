import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn.init

# device 설정
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# 시드 설정
torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)


# parameters
learning_rate = 0.001
epochs = 15
batch_size = 100


# MNIST 데이터
mnist_train = dsets.MNIST(root='MNIST_data/', train=True,
                          transform=transforms.ToTensor(), download=True)
mnist_test = dsets.MNIST(root='MNIST_data/', train=False,
                          transform=transforms.ToTensor(), download=True)


data_loader = torch.utils.data.DataLoader(dataset=mnist_train, batch_size=batch_size,
                                          shuffle=True, drop_last=True)


# CNN 구조 생성
class CNN(torch.nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        self.drop_prob = 0.5

        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        self.fc1 = torch.nn.Linear(3*3*128, 625, bias=True)
        torch.nn.init.xavier_uniform_(self.fc1.weight)

        self.layer4 = torch.nn.Sequential(
            self.fc1,
            torch.nn.ReLU(),
            torch.nn.Dropout(p=self.drop_prob)
        )

        self.fc2 = torch.nn.Linear(625, 10, bias=True)
        torch.nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        output = self.layer1(x)
        output = self.layer2(output)
        output = self.layer3(output)
        output = output.view(output.size(0), -1)
        output = self.layer4(output)
        output = self.fc2(output)
        return output


# 모델 / loss / optimizer 정의
model = CNN().to(device)
criterion = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# 모델 학습
total_batch = len(data_loader)
model.train()
for epoch in range(epochs):
    avg_cost = 0
    for X, Y in data_loader:
        X = X.to(device)
        Y = Y.to(device)

        optimizer.zero_grad()

        hypothesis = model(X)
        loss = criterion(hypothesis, Y)
        loss.backward()
        optimizer.step()

        avg_cost += loss / total_batch

    print('[Epoch: {:>4}] cost = {:>.9}'.format(epoch + 1, avg_cost))

print('learning finished')


with torch.no_grad():
    model.eval()

    X_test = mnist_test.test_data.view(len(mnist_test), 1, 28, 28).float().to(device)
    Y_test = mnist_test.test_labels.to(device)

    prediction = model(X_test)
    correct_prediction = torch.argmax(prediction, 1) == Y_test
    accuracy = correct_prediction.float().mean()
    print('Accuracy : ', accuracy.item())










