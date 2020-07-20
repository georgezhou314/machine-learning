#!/usr/bin/env python
# coding: utf-8

# In[17]:


import torch
import numpy as np
import sys
sys.path.append("..")
import d2lzh_pytorch.utils as d2l


# ## 读取数据

# In[18]:


batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)


# ## 定义模型

# In[19]:


# 图片输入为28*28,输出为10个类别，隐藏曾为256超参数
num_inputs,num_outputs,num_hiddens = 784,10,256

w1 = torch.tensor(np.random.normal(0,0.01,(num_inputs,num_hiddens)),dtype=torch.float)
b1 = torch.zeros(num_hiddens,dtype=torch.float)
w2 = torch.tensor(np.random.normal(0,0.01,(num_hiddens,num_outputs)),dtype=torch.float)
b2 = torch.zeros(num_outputs,dtype=torch.float)

params = [w1,b1,w2,b2]
for param in params:
    param.requires_grad_(requires_grad=True)


# ## 定义激活函数

# In[20]:


def relu(x):
    # max函数返回x对应元素和0的较大值
    return torch.max(input=x,other=torch.tensor(0.0))


# ## 定义模型
# 
# 

# In[21]:


def net(x):
    x = x.view(-1,num_inputs)
    h = relu(torch.matmul(x,w1)+b1)
    return torch.matmul(h,w2)+b2


# ## 定义损失函数

# In[22]:


loss = torch.nn.CrossEntropyLoss()


# ## 训练模型

# In[23]:


num_epochs,lr=5,100.0
# 使用默认的优化函数SGD
d2l.train_ch3(net,train_iter,test_iter,loss,num_epochs,batch_size,params,lr)


# In[ ]:




