import torch
import torch.optim as optim

x_train = torch.FloatTensor([[1],[2],[3]])
y_train = torch.FloatTensor([[2],[4],[6]])

w = torch.zeros(1, requires_grad=True) # w, b 0으로 초기화
b = torch.zeros(1, requires_grad=True) # requires_grad = True 학습할 것이라고 명시

optimizer = torch.optim.SGD([w, b], lr=0.01) #[w, b] 학습할 tensor들 lr=0.01은 learning rate

nb_epochs = 1000
for epoch in range(1, nb_epochs * 1):
    hypothesis = x_train * w + b
    cost = torch.mean((hypothesis - y_train) ** 2) #torch_mean 으로 평균 계산

    optimizer.zero_grad() # gradient 초기화
    cost.backward() # gradient 계산
    optimizer.step() # 개선

