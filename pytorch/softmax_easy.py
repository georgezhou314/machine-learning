#!/usr/bin/env python
# coding: utf-8

# In[35]:


import torch
from torch import nn
from torch.nn import init
import numpy as np
import sys
sys.path.append("..")
import d2lzh_pytorch.utils as d2l


# ## 读取数据

# In[36]:


batch_size = 256
train_iter,test_iter = d2l.load_data_fashion_mnist(batch_size)


# ## 定义模型

# In[37]:


num_inputs = 28*28
num_outputs = 10
# 提供把x矩阵转换为n* (28*28)的二维矩阵
class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer,self).__init__()
    # 这里不经过处理,只是简单的转换一下矩阵的size,原来的size是256*1*28*28,转换为256*(28*28)
    def forward(self,x):
        return x.view(x.shape[0],-1)


# In[38]:


from collections import OrderedDict
net = nn.Sequential(
    OrderedDict([
        ('flatten',FlattenLayer()),
        ('linear',nn.Linear(num_inputs,num_outputs))
    ]
    )
)


# In[39]:


# 模型参数初始化
init.normal_(net.linear.weight, mean=0, std=0.01)
init.constant_(net.linear.bias, val=0)


# ## 定义交叉熵损失函数

# In[40]:


# -log(softmax(x).argmax(dim=1)).sum(),计算交叉熵的方法,先经过softmax处理过后,
# 选择每列最大的元素保留下来,然后使用-log函数处理,最后求和
loss  = nn.CrossEntropyLoss()


# ## 定义优化算法

# In[41]:


optimizer = torch.optim.SGD(net.parameters(),lr=0.1)


# ## 训练模型

# In[42]:


num_epochs = 5
d2l.train_ch3(net,train_iter,test_iter,loss,num_epochs,batch_size,None,None,optimizer)


# In[53]:


# 验证数据据,选出的一张图,得到最大概率的分类,对应label即可查看出是什么类别
x ,y =iter(train_iter).next()
print(nn.functional.softmax(net(x[0]),dim=1).argmax())


# In[ ]:




