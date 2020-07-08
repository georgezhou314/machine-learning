#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import torch
from IPython import display
import matplotlib.pyplot as plt
import numpy as np
import random


# In[26]:


num_inputs = 2
features = torch.randn(1000,num_inputs,dtype=torch.float32)
# 权重
true_w = [2,-3.4]
# 偏差
true_b = 4.2
labels = true_w[0]*features[:,0]+true_w[1]*features[:,1]+true_b
rd = torch.tensor(np.random.normal(0,0.01,size=labels.size()),dtype=torch.float32)
labels += rd


# In[27]:


def use_svg_display():
    display.set_matplotlib_formats('svg')
def set_figsize(figsize=(6.5,4.5)):
    use_svg_display()
    plt.rcParams['figure.figsize'] = figsize
set_figsize()
plt.scatter(features[:,1].numpy(),labels.numpy())


# ## batch_size函数

# In[28]:


def data_iter(batch_size,features,labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0,num_examples,batch_size):
        j = torch.LongTensor(indices[i: min(i+batch_size,num_examples)])
        # 迭代，在0维取出索引位于j的元素，即取出索引为j的二维向量
        yield features.index_select(0,j),labels.index_select(0,j)


# ## 取出批量数据测试

# In[29]:


batch_size = 10
for x,y in data_iter(batch_size,features,labels):
    print(x,y)
    break


# ## 初始化模型参数

# In[36]:


# 生成一个服从(0,0.01)的2*1的weight矩阵
w = torch.tensor(np.random.normal(0,0.01,(num_inputs,1)),dtype=torch.float32)
b = torch.zeros(1,dtype=torch.float32)
w.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True)


# ## 定义模型

# In[37]:


def linreg(x,w,b):
    return torch.mm(x,w)+b


# ## 定义损失函数

# In[39]:


def squared_loss(y_hat,y):
    # 返回的是向量
    return (y_hat-y.view(y_hat.size()))**2/2


# ## 定义优化算法

# In[40]:


def sgd(params,lr,batch_size):
    for param in params:
        param.data -= lr*param.grad/batch_size


# ## 训练模型

# In[49]:


lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss

for epoch in range(num_epochs):
    for x,y in data_iter(batch_size,features,labels):
        l = loss(net(x,w,b),y).sum()
        l.backward()
        sgd([w,b],lr,batch_size)
        # 清零梯度
        w.grad.data.zero_()
        b.grad.data.zero_()
    train_l = loss(net(features,w,b),labels)
    print('epoch %d, loss %f '% (epoch+1,train_l.mean().item()))


# ## 比较真实的参数

# In[47]:


print(true_w,w)
print(true_b,b)


# In[ ]:




