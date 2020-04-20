import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as dsets
import visdom

# visdom에 이미지를 보낼 수 있는 객체 생성
vis = visdom.Visdom()

# # ex1. string 출력해보기
vis.text('hello, world', env='main')
# env 를 같은 이름으로 지정한 vis 객체들을 하나의 명령어로 다같이 종료할 수 있게된다.

# ex2. 하나의 이미지 띄우기
a = torch.randn(3, 200, 200)
vis.image(a)

# ex3. 한번에 여러장의 이미지를 띄우기
vis.images(torch.randn(3, 3, 28, 28)) # 배치, 채널, width, height

# MNIST 와 CIFAR10 이미지를 출력해보기위해 데이터셋 불러오기
MNIST = dsets.MNIST(root="./MNIST_data",train = True,transform=torchvision.transforms.ToTensor(), download=True)
cifar10 = dsets.CIFAR10(root="./cifar10",train = True, transform=torchvision.transforms.ToTensor(),download=True)

# ex4. CIFAR10 에서 하나의 이미지를 출력해보기
data = cifar10.__getitem__(0)
print(data[0].shape)
vis.image(data[0], env='main')

# ex5. MNIST 에서 하나의 이미지를 출력해보기
data = MNIST.__getitem__(0)
print(data[0].shape)
vis.images(data[0], env='main')

# data loader를 통해 batch size만큼 출력해보기
data_loader = torch.utils.data.DataLoader(dataset= MNIST, batch_size=32, shuffle=True)

for numm, value in enumerate(data_loader):
    value = value[0]
    print(value.shape)
    vis.images(value)
    break

# ex6. Line Plot 그리기
y_data = torch.randn(5)
plt = vis.line(Y=y_data)
# X축을 설정하지 않으면 0과 1사이로 출력
# X축을 아래와같이 설정할 수도 있다.
x_data = torch.Tensor([1, 2, 3, 4, 5])
plt = vis.line(Y=y_data, X=x_data)


# ex7. 이미 있는 Line Plot에 값을 업데이트하기
y_append = torch.rand(1)
x_append = torch.Tensor([6])
# 위에서 line plot의 이름을 plt로 정의했고 그곳에 업데이트하려면 win의 이름에 plt를, update는 append 로 설정
vis.line(Y=y_append, X=x_append, win=plt, update='append')

# ex8. 하나의 window에 여러개의 선 그리기
num = torch.Tensor(list(range(0, 10)))
num = num.view(-1, 1)
num = torch.cat((num, num), dim=1)
plt = vis.line(Y=torch.rand(10, 2), X=num)

# ex9. plot에 정보 붙이기
# opts에 dict 형태로 정보를 붙일 수 있다.
plt = vis.line(Y=y_data, X=x_data, opts=dict(title='Test plot', showlegend=True))
# legend를 리스트 형태로 제목을 지정해줄 수 있다.
plt = vis.line(Y=y_data, X=x_data, opts=dict(title='Test plot', legend=['1번'], showlegend=True))
# 두개의 plot에 이름을 붙일 수도 있다.
plt = vis.line(Y=torch.randn(10,2), X=num, opts=dict(title='Test plot', legend=['1번', '2번'], showlegend=True))

# ex10. 함수를 통해서 그래프를 업데이트 시키기.
def loss_tracker(loss_plot, loss_value, num):
    vis.line(X=num,
             Y=loss_value,
             win=loss_plot,
             update='append')

plt = vis.line(Y=torch.Tensor(1).zero_())

for i in range(500):
    loss = torch.randn(1) + i
    loss_tracker(plt, loss, torch.Tensor([i]))

# 아래 명령어로 창을 끌수 있다.
# vis.close(env="main")









