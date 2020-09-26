#!/usr/bin/env python
# coding: utf-8

# In[24]:


'''
一个最简单的识别验证码的程序，验证码经过分割 灰度 二值化处理
使用类似LeNet的结果，识别成功率很高，达到了99.2%
'''
import time
import torch
from torch import nn,optim
import numpy as np
from torchvision import datasets,transforms,utils
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ## 定义网络

# In[25]:


# 注意，这里的图片的大小是1*18*18，不是文中给出的例图的大小
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1,6,5),
            nn.Sigmoid(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(6,16,5),
            nn.Sigmoid(),
            nn.MaxPool2d(2,2)
        )
        self.fc = nn.Sequential(
            nn.Linear(16,10)
            #nn.Linear(16*1*1,120),
            #nn.Sigmoid(),
            #nn.Linear(120,84),
            #nn.Sigmoid(),
            #nn.Linear(84,10)
            
        )
    def forward(self,img):
        # 把三维图片压缩成1维的，由于第2维和第三维都一样
        img = img[0:,0,0:,0:]
        img = img.unsqueeze(dim=1)
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0],-1))
        return output
net = LeNet()
# print(net)
# 检测是否能正常处理图片数据
x = torch.randn(1,3,18,18)
y = net(x)
print(y)


# ## 获取数据训练

# In[26]:


batch_size = 32
# 其中的数据，按照文件夹名字进行分类，例如 目录0下的所有图片都是字符0
train_data = datasets.ImageFolder(r"/home/george/number/split",transform=
                                  transforms.Compose([transforms.ToTensor()]))
test_data = datasets.ImageFolder(r"/home/george/number/split_test",transform=
                                  transforms.Compose([transforms.ToTensor()]))
# [0,1,2,3,4,...9]
# classes =  train_data.classes
# print('classes',len(classes))
train_iter = torch.utils.data.DataLoader(train_data,batch_size=batch_size,shuffle=True)
test_iter = torch.utils.data.DataLoader(test_data,batch_size=batch_size,shuffle=True)


# ### 定义测试评估函数，使用的是d2lzh_pytorch中的定义

# In[27]:


def evaluate_accuracy(data_iter, net, device=None):
    if device is None and isinstance(net, torch.nn.Module):
        # 如果没指定device就使用net的device
        device = list(net.parameters())[0].device 
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(net, torch.nn.Module):
                net.eval() # 评估模式, 这会关闭dropout
                acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
                net.train() # 改回训练模式
            else: # 自定义的模型, 3.13节之后不会用到, 不考虑GPU
                if('is_training' in net.__code__.co_varnames): # 如果有is_training这个参数
                    # 将is_training设置成False
                    acc_sum += (net(X, is_training=False).argmax(dim=1) == y).float().sum().item() 
                else:
                    acc_sum += (net(X).argmax(dim=1) == y).float().sum().item() 
            n += y.shape[0]
    return acc_sum / n


# ## 训练函数

# In[28]:


def train_ch5(net,train_iter,test_iter,batch_size,optimizer,device,num_epochs):
    net = net.to(device)
    print("training on ",device)
    loss = torch.nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        train_l_sum,train_acc_sum,n,batch_count,start = 0.0,0.0,0,0,time.time()
        for x,y in train_iter:
            x = x.to(device)
            y = y.to(device)
            y_hat = net(x)
            l = loss(y_hat,y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        test_acc = evaluate_accuracy(test_iter,net)
        print('epoch %d,loss %.4f,train acc %.3f,test acc %.3f,time %.1f sec' 
              %(epoch+1,train_l_sum/batch_count,train_acc_sum/n,test_acc,time.time()-start))


# In[29]:


lr,num_epochs = 0.01,5
optimizer = torch.optim.Adam(net.parameters(),lr=lr)
train_ch5(net,train_iter,test_iter,batch_size,optimizer,device,num_epochs)


# ### 随便加载一张图，来测试，手动观察是否识别成功

# In[30]:


toy_data = datasets.ImageFolder(r"/home/george/number/manual",transform=
                                  transforms.Compose([transforms.ToTensor()]))
toy_iter = torch.utils.data.DataLoader(toy_data,batch_size=1,shuffle=True)
for img,label in toy_iter:
    img = img.to(device)
    output = net(img)
    print(output.argmax(dim=1).item())


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




