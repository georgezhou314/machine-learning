import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

class Net(nn.Module):
	def __init__(self):
		super(Net,self).__init__()
		self.conv1 = nn.Conv2d(3,8,5)
		self.pool = nn.MaxPool2d(2,2)
		self.conv2 = nn.Conv2d(8,16,5)
		self.fc1 = nn.Linear(16*5*5,120)
		self.fc2 = nn.Linear(120,84)
		self.fc3 = nn.Linear(84,10)
	def forward(self,x):
		# 维度:32-5+1 = 28, 28/2=14
		x = self.pool(F.relu(self.conv1(x)))
		# 维度:14-5+1 = 10, 10/2=5
		x = self.pool(F.relu(self.conv2(x)))
		x = x.view(-1,16*5*5)
		#print("x size:",x.size())
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x
def imshow(img):
	# 由于之前在数据下载时，做了归一化，这里做个相反的操作
	img = img/2 +0.5
	# tensor转换为numpy
	npimg = img.numpy()
	# transpose转换1，2，3维的数据，转换为32*128*3的数据，显示
	plt.imshow(np.transpose(npimg,(1,2,0)))
	plt.show()

# ToTensor()把灰度范围0-255变换为0-1.之后Normalize把0-1变换为(-1,-1);
# Normalize执行,image = (image-mean)/std,其中mean和std依据(0.5,0.5,0.5)和(0.5,0.5,0.5)指定
transform = transforms.Compose([transforms.ToTensor(),
				transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
trainset = torchvision.datasets.CIFAR10(root='./data',train=True,
					download=True,transform=transform)
trainloader = torch.utils.data.DataLoader(trainset,batch_size=4,
					 shuffle=True,num_workers=2)
testset = torchvision.datasets.CIFAR10(root='./data',train=False,
				      download=True,transform=transform)
testloader = torch.utils.data.DataLoader(testset,batch_size=4,shuffle=False,num_workers=2)
classes = ('plane','car','bird','cat','deer','dog','frog','horse','ship','truck')

'''
# 加载图片,一次取出一个batch的数量的图片
dataiter = iter(trainloader)
images,labels = dataiter.next()
# make_grid函数，拼接图片
imshow(torchvision.utils.make_grid(images,padding=0))
# 拼接字符串，'%' 控制格式
print(''.join("%6s" % classes[labels[j]] for j in range(4)))
'''

##### 网络部分
net = Net()
# Loss函数,交叉损失熵
criterion = nn.CrossEntropyLoss()
# 优化，使用随机梯度，lr为学习率，momentum为动量
optimizer = optim.SGD(net.parameters(),lr=0.001,momentum=0.9)
# train the net
for epoch in range(2):
	running_loss = 0.0
	for i,data in enumerate(trainloader,0):
		inputs,labels = data
		optimizer.zero_grad()
		outputs = net(inputs)
		loss = criterion(outputs,labels)
		loss.backward()
		optimizer.step()

		running_loss += loss.item()
		if i%2000 ==1999:
			print("[%d %5d ]loss:%.3f " %(epoch+1,i+1,running_loss/2000))
			running_loss = 0.0
print('Finished Training')
# 保存训练结果
PATH = './cifar_net.pth'
torch.save(net.state_dict(),PATH)

# next()方法返回一个列表，images和labels分别接收第一项和第二项
dataiter = iter(testloader)
images,labels = dataiter.next()
imshow(torchvision.utils.make_grid(images))
print('GroundTruth:',''.join('%5s' %classes[labels[j]] for j in range(4)))
# 加载网络
net = Net()
PATH = './cifar_net.pth'
net.load_state_dict(torch.load(PATH))
outputs = net(images)
print("outputs:",outputs)
# outputs是一个4*10的矩阵，每张图对应一个10维的向量，所以选择一个概率最大
# max(outs,1)意为返回每一行最大的值,predicted为下标,_为最大的值
_,predicted = torch.max(outputs,1)
print('predicted',''.join("%5s" %classes[predicted[j]] for j in range(4)))

# 测试集

correct = 0
total = 0
with torch.no_grad():
	for data in testloader:
		images,labels = data
		outputs = net(images)
		_,predicted = torch.max(outputs,1)
		total += labels.size(0)
		correct += (predicted ==labels ).sum().item()
# %%格式控制，输出'%'
print('Accuracy of the network on the 10000 test images: %d %%' % (100*correct/total))

# 依次查看各个种类的正确率
'''
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
	for data in testloader:
		images,labels = data
		outputs = net(images)
		_,predicted = torch.max(outputs,1)
		c = (predicted == labels).squeeze()
		for i in range(4):
			label = labels[i]
			# 如果预测对了，那么c[i]为True，否则为False
			class_correct[label] += c[i].item()
			class_total[label] += 1
for i in range(10):
	print('Accuracy of %6s : %2d %%' % (classes[i],100*class_correct[i]/class_total[i]))
'''
