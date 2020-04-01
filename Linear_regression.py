import torch

x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[2], [4], [6]])

w = torch.zeros(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

optimizer = torch.optim.SGD([w, b], lr=0.01)

epochs = 1000

for epoch in range(1, epochs+1):
    hypothesis = x_train * w + b
    loss = torch.mean((hypothesis - y_train) ** 2)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(w)
print(b)

x_test = torch.FloatTensor([[4]])

print(x_test * w + b)

