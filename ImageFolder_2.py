import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(777)
if device =='cuda':
    torch.cuda.manual_seed_all(777)

trans = transforms.Compose([
            transforms.ToTensor()
])
# ImageFolder_1 에서 만든 train_data를 trans를 이용해 Tensor로 변환해 가져오기
train_data = torchvision.datasets.ImageFolder(root='data/custom_data/train_data', transform=trans)

data_loader = DataLoader(dataset=train_data, batch_size=8, shuffle=True)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(16*13*29, 120),
            nn.ReLU(),
            nn.Linear(120, 2)
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.shape[0], -1)
        out = self.layer3(out)
        return out

net = CNN().to(device)
# test_tensor = torch.Tensor(3, 3, 64, 128).to(device)
# test_out = net(test_tensor)

optimizer = torch.optim.Adam(net.parameters(), lr=0.0005)
criterion = nn.CrossEntropyLoss().to(device)

total_batch = len(data_loader)

epochs = 7
for epoch in range(epochs):
    avg_cost = 0
    for num, data in enumerate(data_loader):
        imgs, labels = data
        imgs = imgs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        predict = net(imgs)
        loss = criterion(predict, labels)
        loss.backward()
        optimizer.step()

        avg_cost += loss / total_batch
    print('[Epoch : {} , Loss : {}'.format(epoch+1, avg_cost))

print('Learning finished')

# 학습된 weight 저장
torch.save(net.state_dict(), './model/model.pth')

# 저장한 weight를 불러오는 방법
# 정의한 모델의 이름은 그대로 가져오는 새로운 객체 생성
new_net = CNN().to(device)
new_net.load_state_dict(torch.load('./model/model.pth'))


# test_data를 이용해 accuracy 구하기.
trans = torchvision.transforms.Compose([
    transforms.Resize((64, 128)),
    transforms.ToTensor()
])
test_data = torchvision.datasets.ImageFolder(root='data/custom_data/test_data', transform=trans)
# batch size를 test_data 크기만큼해서 한번에 모든 데이터를 입력값으로 넣는다.
test_loader = DataLoader(dataset=test_data, batch_size=len(test_data))

with torch.no_grad():
    for num, data in enumerate(test_loader):
        imgs, labels = data
        imgs = imgs.to(device)
        labels = labels.to(device)

        predict = net(imgs)

        correct_predict = torch.argmax(predict, 1) == labels
        accuracy = correct_predict.float().mean()
        print('Accuracy : ', accuracy.item())













