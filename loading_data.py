import torch
import torch.optim as optim
from torch.utils.data import Dataset      #torch.util.data 상속
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F


class CustomDataset(Dataset):
    def __init__(self):
        self.x_data = [[73, 80, 75],
                        [93, 88, 93],
                        [89, 91, 90],
                        [96, 98, 100],
                        [73, 66, 70]]
        self.y_data =  [[152], [185], [180], [196], [142]]

# 이 데이터셋의 총 데이터 수
    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):   # 어떠한 인덱스 idx를 받았을 때, 그에 상응하는 입출력 데이터 반환
        x = torch.FloatTensor(self.x_data[idx])
        y = torch.FloatTensor(self.y_data[idx])

        return x, y

dataset = CustomDataset()

dataloader = DataLoader(
    dataset,
    batch_size = 2,                 #각 minibatch의 크기
    shuffle=True,                   #epoch마다 데이터셋을 섞어서 데이터가 학습되는 순서를 바꾼다.
)

class MultivariateLinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 1)
    def forward(self, x):
        return self.linear(x)

model = MultivariateLinearRegressionModel()


optimizer = optim.SGD(model.parameters(), lr=1e-5)

nb_epochs = 20
for epoch in range(nb_epochs + 1):
  for batch_idx, samples in enumerate(dataloader):
    x_train, y_train = samples

    prediction = model(x_train)

    cost = F.mse_loss(prediction, y_train)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    print('Epoch {:4d}/{} Batch: {}/{} Cost: {:.6f}'.format(
        epoch, nb_epochs, batch_idx+1, len(dataloader),
        cost.item()
    ))