import torch
from torch import nn
import numpy as np


# 不带偏置的模型
class MF(nn.Module):
    def __init__(self, user_num, item_num):
        super(MF, self).__init__()
        self.user = nn.Embedding(user_num, 2)
        self.item = nn.Embedding(item_num, 2)

    def forward(self, x, y):
        rating = torch.sum(self.user(x) * self.item(y))
        return rating


def train(mf, epochs, optim):
    loss = nn.MSELoss()
    for epoch in range(epochs):
        for u in range(25):
            for i in range(100):
                # 没有评分数据的u-i不参与训练
                if rating_matrix[u][i] == 0:
                    continue
                u_t = torch.tensor(u)
                i_t = torch.tensor(i)
                optim.zero_grad()
                predict = mf(u_t, i_t)
                # 后半部分注释掉的为正则化内容
                l = loss(predict, torch.tensor(rating_matrix[u][i], dtype=torch.float))#+0.1*(torch.norm(mf.user(u_t))+torch.norm(mf.item(i_t)))
                l.backward()
                optim.step()


file_name = 'user_item_rating.txt'
rating_matrix = np.loadtxt(file_name, dtype=bytes).astype(float)
user_num = rating_matrix.shape[0]
item_num = rating_matrix.shape[1]

mf = MF(user_num, item_num)
optimizer = torch.optim.SGD(mf.parameters(), lr=0.01, weight_decay=0.0)
train(mf, 10, optimizer)
# 查看模型的预测分数
user = torch.tensor(np.arange(0, 25))
item = torch.tensor(np.arange(0, 100))
u_i = torch.matmul(mf.user(user), mf.item(item).T)
torch.set_printoptions(profile="full")
print(u_i)
