import torch
from torch import nn
x = torch.rand(1,2,3,3)
# 1*1卷积核
conv = nn.Conv2d(2,3,1)
y = conv(x)
params = list(conv.parameters())
weight = params[0]
bias = params[1]
print("x.shape",x.shape)
print("weight.shape:",weight.shape)
print("bias.shape",bias.shape)
print("output.shape",y.shape)
y_hat  = torch.zeros(1,3,3,3)
lst = []
for i in range(weight.shape[0]):
	ele_1 = weight[i][0].item()
	ele_2 = weight[i][1].item()
	piece = x[0][0]*ele_1+x[0][1]*ele_2+bias[i]
	#piece.unsqueeze(0)
	lst.append(piece)
one = lst[0].unsqueeze(0)
two = lst[1].unsqueeze(0)
three = lst[2].unsqueeze(0)
y_hat = torch.cat((one,two,three),dim=0).unsqueeze(0)
print(y)
print(y_hat)
# 判断内置卷积和我们运算是否相等
print(torch.allclose(y,y_hat))

