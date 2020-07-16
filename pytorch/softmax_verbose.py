#!/usr/bin/env python
# coding: utf-8

# In[32]:


# 本程序將從零實現softmax
import torch
import torchvision
import numpy as np
import sys
sys.path.append("..")
import d2lzh_pytorch.utils as d2l


# ## 獲取和讀取數據

# In[33]:


batch_size = 256
train_iter,test_iter = load_data_fashion_mnist(batch_size)


# ## 初始化模型參數

# In[34]:


## 因爲圖片爲28*28，輸出有10個類別,相當於每個像素點都對10個類別產生一定的權重
num_inputs = 28*28
num_outputs = 10
w = torch.tensor(np.random.normal(0,0.01,(num_inputs,num_outputs)),dtype=torch.float)
b = torch.zeros(num_outputs,dtype=torch.float)
w.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True)


# ## 定義softmax

# In[35]:


def softmax(x):
    x_exp = x.exp()
    partition = x_exp.sum(dim=1,keepdim=True)
    return x_exp/partition


# In[36]:


softmax(torch.rand(2,5))


# ## 定義模型

# In[37]:


def net(x):
    return softmax(torch.mm(x.view(-1,num_inputs),w)+b)


# In[38]:


## 定義交叉熵

gater函數，這裏應該是爲了挑選出一個矩陣中的目標元素，按照維度給的1，那麼根據所給的index矩陣按照列來調整。

# ## 定义交叉熵损失函数

# In[39]:


def cross_entropy(y_hat,y):
    # 这里y为index矩阵,y调整为n*1的矩阵,意味着每一行只能选择一个元素,这里softmax选一个最大的
    return -torch.log(y_hat.gather(1,y.view(-1,1)))


# ## 定义计算分类准确率

# In[40]:


# 计算预测正确的比率
def evaluate_accuracy(data_iter,net):
    acc_sum,n = 0.0,0
    for x,y in data_iter:
        # 因为有多个样本,所以要用sum加在一起
        acc_sum += (net(x).argmax(dim=1) == y).float().sum().item()
        # y.shape[0]为行数,即为样本数量,因为y的实际参数是[5,2,1,0,9,...,0]这样的类型,
        # 所以y.shape[0]等同于样本的数量
        n += y.shape[0]
    return acc_sum/n


# In[41]:


print(evaluate_accuracy(test_iter,net))


# ## 训练模型

# In[52]:


num_epochs,lr = 5,0.1
#
def train_ch3(net,train_iter,test_iter,loss,num_epochs,batch_size,
              params=None,lr=None,optimizer=None):
    for epoch in range(num_epochs):
        train_l_sum,train_acc_sum,n = 0.0,0.0,0
        for x,y in train_iter:
            y_hat = net(x)
            # 因为一批样本有很多个,所以要计算sum,前面交叉熵损失函数使用-log是因为每个元素都小于1, 
            # 这样能得到正数,再者越接近1, 得到的越精确,loss也越小
            l = loss(y_hat,y).sum()
            # 清零梯度
            if optimizer is not None:
                optimizer.zero_grad()
            elif params is not None and params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()
            # 计算梯度
            # print(b.grad)
            # 每次使用完backward都会自动更新梯度,然后使用梯度,再来更新参数
            l.backward()
            if optimizer is None:
                d2l.sgd(params,lr,batch_size)
            else:
                optimizer.step()
            # train_loss_sum
            train_l_sum += l.item()
            # train_accumlate_sum,代表训练的正确率
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
            n += y.shape[0]
        test_acc = evaluate_accuracy(test_iter,net)
        print('epoch %d, loss %.4f, train acc: %.3f, test acc %.3f'
             % (epoch+1, train_l_sum/n,train_acc_sum/n,test_acc))
# 这里没有选择optimizer
train_ch3(net,train_iter,test_iter,cross_entropy,num_epochs,batch_size,[w,b],lr)


# ## 预测

# In[101]:


# 这里x,y一次性取出256个样本
x,y = iter(test_iter).next()
true_labels = d2l.get_fashion_mnist_labels(y.numpy())
pred_labels = d2l.get_fashion_mnist_labels(net(x).argmax(dim=1).numpy()) 

# title为两行,上面是真实的,下面是预测的,竟然完全一样
titles = [true + '\n' + pred for true,pred in zip(true_labels,pred_labels)]
d2l.show_fashion_mnist(x[0:10],titles[0:10])

# 随便生成一些图,做测试
rd = torch.randn(10,28,28)
pred_labels = d2l.get_fashion_mnist_labels(net(rd).argmax(dim=1).numpy()) 
d2l.show_fashion_mnist(rd,pred_labels)


# In[ ]:





# In[ ]:




