import torch
from torch import nn
import numpy as np


# 带偏置的模型, 其中b_u和b_i也是可学习的参数
class MF(nn.Module):
    def __init__(self, user_num, item_num, miu):
        super(MF, self).__init__()
        self.user = nn.Embedding(user_num, 2)
        self.item = nn.Embedding(item_num, 2)
        self.b_u = nn.Embedding(user_num, 1)
        self.b_i = nn.Embedding(item_num, 1)
        self.miu = miu

    def forward(self, x, y):
        rating = torch.matmul(self.user(x), self.item(y).T)+self.b_u(x)+self.b_i(y)+self.miu
        return rating


def train(mf, epochs, optim):
    loss = nn.MSELoss()
    for epoch in range(epochs):
        loss_sum = 0.0
        cnt = 0
        for u in range(25):
            for i in range(100):
                # 没有评分数据的u-i不参与训练
                if rating_matrix[u][i] == 0:
                    continue
                u_t = torch.tensor(u)
                i_t = torch.tensor(i)
                optim.zero_grad()
                predict = mf(u_t, i_t)
                ground_truth = torch.tensor(rating_matrix[u][i], dtype=torch.float)
                ground_truth = torch.unsqueeze(ground_truth, dim=0)
                # 后半部分注释掉的为正则化内容
                l = loss(predict, ground_truth)+0.0
                l.backward()
                optim.step()
                loss_sum += l
                cnt += 1
        print("epoch%d loss:%.2f" % (epoch, loss_sum/cnt))


file_name = 'user_item_rating.txt'
rating_matrix = np.loadtxt(file_name, dtype=bytes).astype(float)
sum_rating = 0.0
cnt = 0
for i in range(rating_matrix.shape[0]):
    for j in range(rating_matrix.shape[1]):
        if rating_matrix[i][j] != 0:
            cnt += 1
            sum_rating += rating_matrix[i][j]
mean = sum_rating/cnt
user_cnt = rating_matrix.shape[0]
item_cnt = rating_matrix.shape[1]
mf_bias = MF(user_cnt, item_cnt, mean)
optimizer = torch.optim.SGD(mf_bias.parameters(), lr=0.001)
train(mf_bias, 100, optimizer)

# 进行测试
user = torch.tensor(np.arange(0, 25))
item = torch.tensor(np.arange(0, 100))
# 主要是为了维度对齐
u_i = torch.matmul(mf_bias.user(user), mf_bias.item(item).T)+mf_bias.b_u(user).repeat(1, 100)+mf_bias.b_i(item).T.repeat(25,1)+mf_bias.miu
torch.set_printoptions(profile="full")
print(u_i)