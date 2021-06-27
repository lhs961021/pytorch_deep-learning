import torch
import torch.optim as optim

x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[1], [2], [3]])

w = torch.zeros(1) # 모델 초기화
lr = 0.1 # Learining rate 설정

nb_epochs = 10
for epoch in range(1, nb_epochs + 1):
    hypothesis = x_train * w # H(x) 계산
    cost = torch.mean((hypothesis - y_train)**2) #cost gradient 계산
    gradient = torch.sum((w * x_train - y_train) * x_train)
    print('Epoch {:4d}/{} w: {:.3f}, cost: {:.6f}'.format(epoch, nb_epochs, w.item(), cost.item()))
  
    w -= lr * gradient # cost gradient로 H(x) 개선