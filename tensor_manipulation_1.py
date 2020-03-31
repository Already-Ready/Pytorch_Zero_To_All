import torch
import numpy as np

print("--------Q1---------")
# basic 2D array with pytorch

t = torch.FloatTensor([[1., 2., 3.],
                      [4., 5., 6.],
                      [7., 8., 9.],
                      [10., 11., 12.]
                      ])


print(t)

print(t.dim())
print(t.size())
print(t[:, 1])
print(t[:, :-1])


print("--------Q2---------")
# Broadcasting --> 벡터간의 사이즈가 안맞을때 자동 맞춤이 어떻게 될까?

# 두 텐서의 모양이 같을때
m1 = torch.FloatTensor([[3, 3]])
m2 = torch.FloatTensor([[2, 2]])
print(m1 + m2)

# 벡터와 스칼라를 연산할 때
m1 = torch.FloatTensor([[1, 2]])
m2 = torch.FloatTensor([3])
print(m1 + m2) # m2 벡터의 shape이 (1,) 에서 (1,2)로 확장된다.

# 두 텐서의 모양이 다를 때
m1 = torch.FloatTensor([[1, 2]])
m2 = torch.FloatTensor([[3], [4]])
print(m1 + m2)
# m1 이 [1 2] 로 확장되고 원래 [3   ] 이던 m2가 [3 3] 로 확장된다.
#       [1 2]                 [4   ]           [4 4]

print("--------Q3---------")
# 행렬곱(Matrix Multiplication)

m1 = torch.FloatTensor([[1, 2], [3, 4]])
m2 = torch.FloatTensor([[1], [2]])

print("Matrix Multiplication : ", m1.matmul(m2))

print("--------Q4---------")
# 합과 평균

t = torch.FloatTensor([[1, 2], [3, 4]])
print(t)

print("About Sum")
print(t.sum())
print(t.sum(dim=0)) # dim = 0 이란 열을 기준으로 평균을 구하겠다는것
print(t.sum(dim=1)) # dim = 1 이란 행을 기준으로 평균을 구하겠다는것

print("About Mean")
print(t.mean())
print(t.mean(dim=0)) # dim = 0 이란 열을 기준으로 평균을 구하겠다는것
print(t.mean(dim=1)) # dim = 1 이란 행을 기준으로 평균을 구하겠다는것


print("--------Q5---------")
# Max & Argmax

t = torch.FloatTensor([[4, 1], [2, 3]])
print(t)

print(t.max()) # max() 함수안에 dim을 지정해주지 않으면 하나의 값을 리턴한다.

print(t.max(dim=0)) # 열에 대해서 max를 구하라고 dim을 지정하면 values 와 indices 두 가지 값을 리턴한다.
# values는 max값을, indices는 해당 value의 위치를 나타낸다.

print(t.max(dim=1)) # 행에 대한 결과

print("--------Q6---------")

# View (numpy의 reshape)

t  = np.array([[[0, 1, 2],
               [3, 4, 5]],

              [[6, 7, 8],
               [9, 10, 11]]])

ft = torch.FloatTensor(t)
print(ft)
print(ft.shape)

print(ft.view([-1,3]))
print(ft.view([-1,3]).shape)

print(ft.view([-1, 1, 3]))
print(ft.view([-1, 1, 3]).shape)

# ft의 첫 shpae (2,2,3) 의 각 숫자의 곱은 12이다.
# 이 곱한 수가 항상 유지되게 view를 통해 이동하게 된다.
# 첫번째 view를 통해 (4,3)인 행렬을 얻었는데 이를 살펴보면 각 숫자의 곱이 역시 12임을 알 수 있다.
# 두번째 view를 통해 (4,1,3)인 텐서를 얻었는데 이 또한 각 숫자의 곱이 12임을 알 수 있다.
# -1 을 통해 어떤 값이 들어갈지 모르겠다고 말했으나 사실 각 숫자의 곱이 항상 유지되게 reshape되는것을 볼 수 있다.

print("--------Q7---------")
# Squeeze & Unsqueeze

# Squeeze
print("Squeeze")
ft = torch.FloatTensor([[0], [1], [2]])
print(ft)
print(ft.shape) #>>> (3,1)

print(ft.squeeze())
print(ft.squeeze().shape) #>>> (3)

# 차원의 값이 1인 차원을 없애주게 된다.
# 위에서 첫번째 차원 [   ] 안에 3개의 값이 들어가있고 두 번째 차원에는 각각 하나의 값만 들어가 있다.
# 따라서 두번째 차원(차원값이 1인)이 squeeze를 통해 사라지고 첫번째 차원만 남게 된다.

# Unsqueeze
# squeeze와 반대로 지정한 차원에 1을 넣어주게된다.
print("Unsqueeze")
ft = torch.FloatTensor([0, 1, 2])
print(ft.shape) #>>> (3)

print(ft.unsqueeze(dim=0))          # 첫번째 dim에 1을 넣어라
print(ft.unsqueeze(dim=0).shape) #>>> (1,3)
# 이는 view를 통해서도 똑같이 만들 수 있다.
print(ft.view(1,-1))
print(ft.view(1,-1).shape) #>>> (1,3)

print(ft.unsqueeze(dim=1))
print(ft.unsqueeze(dim=1).shape) #>>> (3,1)

print(ft.unsqueeze(dim=-1))         # 마지막 dim에 1을 넣어라 ==> 마지막 dim = 두번째 dim
print(ft.unsqueeze(dim=-1).shape) #>>> (3,1)

print("--------Q8---------")
# type casting

lt = torch.LongTensor([1, 2, 3, 4])
print(lt)

print(lt.float())

bt = torch.ByteTensor([True, False, False, True])
print(bt)

print(bt.long())
print(bt.float())

print("--------Q9---------")
# Concatenate

x = torch.FloatTensor([[1, 2],
                       [3, 4]]) #>>> (2,2)
y = torch.FloatTensor([[5, 6],
                       [7, 8]]) #>>> (2,2)

print(torch.cat([x,y], dim=0)) #>>> (4,2)
print(torch.cat([x,y], dim=1)) #>>> (2,4)

print("--------Q10---------")
# stack

x = torch.FloatTensor([1, 4]) #>>> (2)
y = torch.FloatTensor([2, 5])
z = torch.FloatTensor([3, 6])

print(torch.stack([x,y,z])) #>>>(3,2) 첫번째 dim에 합해지는 갯수만큼 쌓이게된다.
print(torch.stack([x,y,z], dim=1)) #>>>(2,3) 두번째 dim에 합해지는 갯수만큼 쌓이게된다

#첫번째 stack은 cat을 통해서도 나타낼 수 있다.
print(torch.cat([x.unsqueeze(0), y.unsqueeze(0), z.unsqueeze(0)], dim=0))
# unsqueeze를 통해 (2) 가 (1,2) 로 바뀌고 cat dim=0 를 통해 (3,2)로 바뀜

print("--------Q11---------")
# ones & zeros

x = torch.FloatTensor([[0, 1, 2], [2, 1, 0]])
print(x)

print(torch.ones_like(x)) # x와 같은 shape이고 1로 가득찬 텐서
print(torch.zeros_like(x)) # x와 같은 shape이고 0으로 가득찬 텐서
# ones_like 와 zeros_like는 같은 device 에 선언해준다.

print("--------Q12---------")
# In-place Operation

x = torch.FloatTensor([[1, 2], [3, 4]])

print(x.mul(2.))
print(x)
print(x.mul_(2.)) # 언더바가 붙게되면 메모리에 새로 선언하지 않고 기존 텐서에 값을 넣겠다는 뜻이다.
print(x) # 2가 곱해진 x가 출력되게 되는것을 확인할 수 있다.
