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


