import torch

x_train = torch.FloatTensor([[73, 80, 75],
                             [93, 88, 83],
                             [89, 91, 90],
                             [96,98, 100],
                             [73, 66, 70]])
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])

w = torch.zeros((3,1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

optimizer = torch.optim.SGD([w,b], lr=1e-5)

epochs = 20

for epoch in range(epochs+1):

    hypothesis = x_train.matmul(w) + b

    loss = torch.mean((hypothesis - y_train) ** 2)

    optimizer.zero_grad
    loss.backward()
    optimizer.step()

    print('Epoch {:4d}/{} hypothesis : {} Cost : {:.6f}'. format(
        epoch, epochs, hypothesis.squeeze().detach(), loss.item()
    ))