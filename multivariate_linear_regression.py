import torch
import torch.optim as optim

x_train = torch.FloatTensor([[73, 80, 75],
                            [93, 88, 93],
                            [89, 91, 90],
                            [96, 98, 100],
                            [73, 66, 70]])
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])

# 모델 초기화
w = torch.zeros((3, 1), requires_grad=True) 
b = torch.zeros(1, requires_grad=True)

# optimizer 정의
optimizer = optim.SGD([w, b], lr=1e-5)

nb_epochs = 20
for epoch in range(nb_epochs + 1):
    hypothesis = x_train.matmul(w) + b # H(x) 계산
    cost = torch.mean((hypothesis - y_train)**2) #cost 계산 (MSE)

    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    print('Epoch {:4d}/{} hypothesis: {} cost: {:.6f}'.format(epoch, nb_epochs, hypothesis.squeeze().detach(), cost.item()))

# 점점 작아지는 cost, 점점 y에 가까워지는 H(x), Learning rate에 따라 발산할수도 있음!