import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class MultivariateLinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 1)

    def forward(self, x):
        return self.linear(x)



class MyDataset(Dataset):

    def __init__(self):
        self.x_train = [[73, 80, 75],
                         [93, 88, 83],
                         [89, 91, 90],
                         [96,98, 100],
                         [73, 66, 70]]
        self.y_train = [[152], [185], [180], [196], [142]]

    def __len__(self):
        return len(self.x_train)

    def __getitem__(self, idx):
        x = torch.FloatTensor(self.x_train[idx])
        y = torch.FloatTensor(self.y_train[idx])

        return x, y

model = MultivariateLinearRegressionModel()
optimizer = optim.SGD(model.parameters(), lr=1e-5)
dataset = MyDataset()
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

epochs = 20

for epoch in range(epochs + 1):
    for batch_idx, train in enumerate(dataloader):
        xt, yt = train

        prediction = model(xt)

        loss = F.mse_loss(prediction, yt)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print('Epoch {:4d}/{} Batch : {}/{} Cost : {:.6f}'.format(
            epoch, epochs, batch_idx+1, len(dataloader), loss.item()
        ))