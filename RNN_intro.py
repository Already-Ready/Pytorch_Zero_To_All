import torch
import numpy as np

input_size = 4
hidden_size = 2

# one hot 벡터로 표현
h = [1,0,0,0]
e = [0,1,0,0]
l = [0,0,1,0]
o = [0,0,0,1]

# input size는 (3,5,4) 가 된다
# 3 --> batch_size
# 5 --> sequence length : 문자열의 길이 = RNN sequence의 길이
# 4 --> input_size : one hot 벡터의 길이 --> embedding 을 한다면 embedding 크기가 될것
input_data_np = np.array([[h,e,l,l,o],
                          [e,o,l,l,l],
                          [l,l,e,e,l]], dtype=np.float32)

input_data = torch.Tensor(input_data_np)

rnn = torch.nn.RNN(input_size, hidden_size)

# output size는 2이므로 rnn셀을 통과한 output은 (3,5,2) shape을 가질것이고 이를 확인해본다.
output, _status = rnn(input_data)

print(output)
print(output.shape)