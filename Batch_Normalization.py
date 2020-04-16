import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import random
from torch.utils.data import Dataset, DataLoader
import matplotlib.pylab as plt

# device and seed configure
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# for reproducibility
random.seed(777)
torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)

# parameters
learning_rate = 0.001
epochs = 10
batch_size = 32


# dataset
mnist_train = dsets.MNIST(root='MNIST_data/', train=True,
                          transform=transforms.ToTensor(), download=True)
mnist_test = dsets.MNIST(root='MNIST_data/', train=False,
                          transform=transforms.ToTensor(), download=True)

train_loader = torch.utils.data.DataLoader(dataset=mnist_train,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          drop_last=True)

test_loader = torch.utils.data.DataLoader(dataset=mnist_test,
                                          batch_size=batch_size,
                                          shuffle=False,
                                          drop_last=True)


# BatchNorm으로 모델을 만들기 위한 nn Linear layer 만들기
linear1 = torch.nn.Linear(784, 32, bias=True)
linear2 = torch.nn.Linear(32, 32, bias=True)
linear3 = torch.nn.Linear(32, 10, bias=True)
relu = torch.nn.ReLU()
bn1 = torch.nn.BatchNorm1d(32)
bn2 = torch.nn.BatchNorm1d(32)

# BatchNorm을 사용하지 않는 모델을 만들기 위한 Linear layer 만들기
nn_linear1 = torch.nn.Linear(784, 32, bias=True)
nn_linear2 = torch.nn.Linear(32, 32, bias=True)
nn_linear3 = torch.nn.Linear(32, 10, bias=True)


# 모델 정의
bn_model = torch.nn.Sequential(linear1, bn1, relu,
                               linear2, bn2, relu,
                               linear3).to(device)
nn_model = torch.nn.Sequential(linear1, relu,
                               linear2, relu,
                               linear3).to(device)


# loss / optimizer 정의
criterion = torch.nn.CrossEntropyLoss().to(device)
bn_optimizer = torch.optim.Adam(bn_model.parameters(), lr=learning_rate)
nn_optimizer = torch.optim.Adam(nn_model.parameters(), lr=learning_rate)


# loss 와 Acc 저장할 리스트 생성(추후 그래프로 나타내기 위해)
train_losses = []
train_Accs = []

valid_losses = []
valid_Accs = []


# 학습 & 각epoch 마다 평가
train_total_batch = len(train_loader)
test_total_batch = len(test_loader)
for epoch in range(epochs):
    bn_model.train()
    for X, Y in train_loader:
        X = X.view(-1, 28*28).to(device)
        Y = Y.to(device)

        # BatchNorm 모델 학습
        bn_prediction = bn_model(X)
        bn_loss = criterion(bn_prediction, Y)
        bn_optimizer.zero_grad()
        bn_loss.backward()
        bn_optimizer.step()

        # Non BatchNorm 모델 학습
        nn_prediction = nn_model(X)
        nn_loss = criterion(nn_prediction, Y)
        nn_optimizer.zero_grad()
        nn_loss.backward()
        nn_optimizer.step()

    # 평가
    with torch.no_grad():
        bn_model.eval()

        # train셋을 통한 평가
        bn_loss, nn_loss, bn_Acc, nn_Acc = 0, 0, 0, 0
        for i, (X, Y) in enumerate(train_loader):
            X = X.view(-1, 28 * 28).to(device)
            Y = Y.to(device)

            # BatchNorm
            bn_prediction = bn_model(X)
            bn_correct_prediction = torch.argmax(bn_prediction, 1) == Y
            bn_loss += criterion(bn_prediction, Y)
            bn_Acc += bn_correct_prediction.float().mean()

            # Non BatchNorm
            nn_prediction = nn_model(X)
            nn_correct_prediction = torch.argmax(nn_prediction, 1) == Y
            nn_loss += criterion(nn_prediction, Y)
            nn_Acc += nn_correct_prediction.float().mean()

        bn_loss = bn_loss / train_total_batch
        nn_loss = nn_loss / train_total_batch
        bn_Acc = bn_Acc / train_total_batch
        nn_Acc = nn_Acc / train_total_batch

        # 하나의 epoch에서 평가한 loss/Acc 저장
        train_losses.append([bn_loss, nn_loss])
        train_Accs.append([bn_Acc, nn_Acc])

        print(
            '[Epoch %d by TRAIN] BatchNorm Loss(Acc) --> bn_loss : %.5f(bn_Acc : %.2f) vs Non BatchNorm Loss(Acc) --> nn_loss : %.5f(nn_acc : %.2f)'
            % ((epoch+1), bn_loss.item(), bn_Acc.item(), nn_loss.item(), nn_Acc.item())
            )

        # test셋을 통한 평가
        bn_loss, nn_loss, bn_Acc, nn_Acc = 0, 0, 0, 0
        for i, (X, Y) in enumerate(test_loader):
            X = X.view(-1, 28 * 28).to(device)
            Y = Y.to(device)

            # BatchNorm
            bn_prediction = bn_model(X)
            bn_correct_prediction = torch.argmax(bn_prediction, 1) == Y
            bn_loss += criterion(bn_prediction, Y)
            bn_Acc += bn_correct_prediction.float().mean()

            # Non BatchNorm
            nn_prediction = nn_model(X)
            nn_correct_prediction = torch.argmax(nn_prediction, 1) == Y
            nn_loss += criterion(nn_prediction, Y)
            nn_Acc += nn_correct_prediction.float().mean()

        bn_loss = bn_loss / test_total_batch
        nn_loss = nn_loss / test_total_batch
        bn_Acc = bn_Acc / test_total_batch
        nn_Acc = nn_Acc / test_total_batch

        # 하나의 epoch에서 평가한 loss/Acc 저장
        valid_losses.append([bn_loss, nn_loss])
        valid_Accs.append([bn_Acc, nn_Acc])

        print(
            '[Epoch %d by VALID] BatchNorm Loss(Acc) --> bn_loss : %.5f(bn_Acc : %.2f) vs Non BatchNorm Loss(Acc) --> nn_loss : %.5f(nn_acc : %.2f)'
            % ((epoch+1), bn_loss.item(), bn_Acc.item(), nn_loss.item(), nn_Acc.item())
            )


def plot_compare(loss_list:list, ylim=None, title=None) -> None:
    bn = [i[0] for i in loss_list]
    nn = [i[1] for i in loss_list]

    plt.figure(figsize=(15, 10))
    plt.plot(bn, label='With BN')
    plt.plot(nn, label='Without BN')

    if ylim:
        plt.ylim(ylim)

    if title:
        plt.title(title)

    plt.legend()
    plt.grid('on')
    plt.show()

plot_compare(train_losses, title='Training Loss At Epoch')
plot_compare(train_Accs, [0, 1.0], title='Training Acc At Epoch')
plot_compare(valid_losses, title='Validation Loss At Epoch')
plot_compare(valid_Accs, [0, 1.0], title='Validation Acc At Epoch')

















