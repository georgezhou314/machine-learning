#!/usr/bin/env python
# coding: utf-8

# ## 应用pyTorch内置功能，简单实现线性回归

# ## 生成数据集

# In[72]:


import torch
import numpy as np
num_inputs = 2
num_examples = 1000
true_w = [2,-3.4]
true_b = 4.2
features = torch.tensor(np.random.normal(0,1,(num_examples,num_inputs)),dtype=torch.float)
labels = true_w[0]*features[:,0]+true_w[1]*features[:,1]+true_b
labels += torch.tensor(np.random.normal(0,0.01,size = labels.size()),dtype=torch.float)


# ## 读取数据

# In[73]:


import torch.utils.data as Data
batch_size = 10
dataset = Data.TensorDataset(features,labels)
data_iter = Data.DataLoader(dataset,batch_size,shuffle = True)


# ## 使用pyTorch中的nn.Model定义线性回归

# In[80]:


import torch.nn as nn
# 两种写法都可以
'''
class LinearNet(nn.Module):
    def __init__(self,n_feature):
        super(LinearNet,self).__init__()
        # 两个参数分别为输入size与输出size
        self.linear = nn.Linear(n_feature,1)
    # 前向传播
    def forward(self,x):
        y = self.linear(x)
        return y
net = LinearNet(num_inputs)
'''
net = nn.Sequential(nn.Linear(num_inputs,1))


# ## 初始化参数

# In[75]:


from torch.nn import init

init.normal_(net[0].weight,mean = 0,std=0.01)
init.constant_(net[0].bias,val=0)


# ## 定义损失函数

# In[76]:


loss = nn.MSELoss()


# ## 定义优化算法

# In[77]:


import torch.optim as optim

optimizer = optim.SGD(net.parameters(),lr=0.03)
print(optimizer)


# ## 训练模型

# In[78]:


num_epochs = 3
for epoch in range(1,num_epochs+1):
    for x,y in data_iter:
        output = net(x)
        l = loss(output,y.view(-1,1))
        # 梯度清零
        optimizer.zero_grad()
        # 知乎看到的，loss.backward()获取所有参数的梯度，optim.step()根据梯度，更新参数
        l.backward()
        optimizer.step()
    print('epoch %d, loss: %f' %(epoch,l.item()))


# In[79]:


# 生成的参数可与预设置的(true_w,true_b)进行对比，非常接近,
for param in net.parameters():
    print(param)


# In[ ]:




