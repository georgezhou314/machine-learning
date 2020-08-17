#!/usr/bin/env python
# coding: utf-8

# ## 二维互相关运算

# In[5]:


import torch
from torch import nn

#x为输入,k为kernel 卷积核
def corr2d(x,k):
    h,w = k.shape
    # 构造y的size
    y = torch.zeros(x.shape[0]-h+1,x.shape[1]-w+1)
    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            y[i,j] = (x[i:i+h,j:j+w]*k).sum()
    return y


# In[6]:


x = torch.tensor([[0,1,2],[3,4,5],[6,7,8]])
k = torch.tensor([[0,1],[2,3]])
corr2d(x,k)


# ## 实现二维卷积层

# In[8]:


class Conv2D(nn.Module):
    def __init__(self,kernel_size):
        super(Conv2D,self).__init__()
        self.weight = nn.Parameter(torch.randn(kernel_size))
        self.bias = nn.Parameter(torch.randn(1))
    def forward(self,x):
        return corr2d(x,self.weight)+self.bias


# ## 检测图像边缘

# In[12]:


# 自己构建一张图片
x = torch.ones(6,8)
x[:,2:6] = 0
print(x)
# 构建一个1×2的卷积核心，当互相关运算时，如果横向相邻元素相同则0，否则非0
k = torch.tensor([[1,-1]])
y = corr2d(x,k)
print(y)


# ## 通过数据学习核数组

# In[19]:


conv2d = Conv2D(kernel_size=(1,2))
step = 20
lr =0.01
for i in range(step):
    y_hat = conv2d(x)
    l = ((y_hat -y )**2).sum()
    l.backward()
    
    # 梯度下降
    conv2d.weight.data -= lr*conv2d.weight.grad
    conv2d.bias.data -= lr*conv2d.bias.grad
    # 梯度清零
    conv2d.weight.grad.fill_(0)
    conv2d.bias.grad.fill_(0)
    if (i+1)%5 == 0:
        print("step %d , loss %.3f" %(i+1,l.item()))


# In[21]:


# 查看卷积核的数组
print(conv2d.weight,conv2d.bias)


# In[ ]:




