import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 시드 고정
torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)

# XOR 데이터
X = torch.FloatTensor([[0, 0], [0, 1], [1, 0], [1, 1]]).to(device)
Y = torch.FloatTensor([[0], [1], [1], [0]]).to(device)

# 4개의 레이어가 있는 MLP(Multi Layer Perceptron)
linear1 = torch.nn.Linear(2, 10, bias=True)
linear2 = torch.nn.Linear(10, 10, bias=True)
linear3 = torch.nn.Linear(10, 10, bias=True)
linear4 = torch.nn.Linear(10, 1, bias=True)
sigmoid = torch.nn.Sigmoid()

# 모델 정의
model = torch.nn.Sequential(linear1, sigmoid, linear2, sigmoid, linear3, sigmoid,
                            linear4, sigmoid).to(device)

#  loss / optimizer 정의
criterion = torch.nn.BCELoss().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=1)

# 학습
for epoch in range(10001):
    optimizer.zero_grad()
    hypothesis = model(X)

    loss = criterion(hypothesis, Y)
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(epoch, loss.item())

# 학습 후 실제값과 예측값 비교해보기
with torch.no_grad():
    hypothesis = model(X)
    predict = (hypothesis > 0.5).float()
    accuracy = (predict == Y).float().mean()

    print('Hyptthesis : ', hypothesis.detach().cpu().numpy(),
          '\nCorrect : ',predict.detach().cpu().numpy(),
          '\nAccuracy : ',accuracy.item())
