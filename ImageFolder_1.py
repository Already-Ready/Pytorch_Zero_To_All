import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from matplotlib.pyplot import imshow

# 예시 이미지의 사이즈가 너무 크기때문에 사이즈 조정을 해줘야한다.
trans = transforms.Compose([
            transforms.Resize((64, 128))
])

train_data = torchvision.datasets.ImageFolder(root='data/custom_data/origin_data', transform=trans)

for num, value in enumerate(train_data):
    data, label = value
    print(data, label)

    # Resize된 이미지들을 라벨값에 따라 gray폴더와 red폴더에 나눠서 담아준다.
    if label == 0:
        data.save('data/custom_data/train_data/gray/%d_%d.jpeg'%(num, label))
    else:
        data.save('data/custom_data/train_data/red/%d_%d.jpeg'%(num, label))