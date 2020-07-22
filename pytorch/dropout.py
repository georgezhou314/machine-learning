#!/usr/bin/env python
# coding: utf-8

# ## 丢弃法，应对过拟合

# In[12]:


get_ipython().run_line_magic('matplotlib', 'inline')
import torch
import torch.nn as nn
import numpy as np
import sys
sys.path.append("..")
import d2lzh_pytorch.utils as d2l
# 第二个参数是丢弃概率，是一个超参数
def dropout(x,drop_prob):
    x = x.float()
    assert 0 <= drop_prob <=1
    keep_prob = 1 - drop_prob
    #
    if keep_prob == 0:
        return torch.zeros_like(x)
    mask = (torch.randn(x.shape)<keep_prob).float()
    # 丢弃法不改变原来的期望值，所以要拉伸，即处以keep_prob
    return mask*x/keep_prob


# ## 定义模型，使用soft-max分类fashion-Mnist

# In[18]:


# 两个隐藏层，一层输入，一层输出
num_inputs,num_outputs,num_hiddens1,num_hiddens2 = 784,10,256,256

w1 = torch.tensor(np.random.normal(0,0.01,size=(num_inputs,num_hiddens1)),dtype=torch.float)
b1 = torch.zeros(num_hiddens1)
w2 = torch.tensor(np.random.normal(0,0.01,size=(num_hiddens1,num_hiddens2)),dtype=torch.float)
b2 = torch.zeros(num_hiddens2)
w3 = torch.tensor(np.random.normal(0,0.01,size=(num_hiddens2,num_outputs)),dtype=torch.float)
b3 = torch.zeros(num_outputs)
params = [w1,b1,w2,b2,w3,b3]
for param in params:
    param.requires_grad_(requires_grad=True)


# In[19]:


# 定义网络
drop_prob1,drop_prob2 = 0.2,0.5
def net(x,is_trainning=True):
    x = x.view(-1,num_inputs)
    h1 = (torch.matmul(x,w1)+b1).relu()
    if is_trainning:
        h1 = dropout(h1,drop_prob1)
    h2 = (torch.matmul(h1,w2)+b2).relu()
    if is_trainning:
        h2 = dropout(h2,drop_prob2)
    return torch.matmul(h2,w3)+b3


# ## 训练&测试`

# In[20]:


num_epochs,lr,batch_size = 5,100.0,256
loss = torch.nn.CrossEntropyLoss()
train_iter,test_iter = d2l.load_data_fashion_mnist(batch_size)
d2l.train_ch3(net,train_iter,test_iter,loss,num_epochs,batch_size,params,lr)


# In[ ]:




