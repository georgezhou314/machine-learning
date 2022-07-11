# 设置可用显卡
import os
import torch
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
torch.cuda.set_device(1)

# 查看当前设备环境可用显卡数量
ng = torch.cuda.device_count()
print("Devices:%d" %ng)

# 3、查看可用显卡的具体信息（型号、算力，显存以及线程数）
print(torch.cuda.get_device_properties(0))
print(torch.cuda.get_device_properties(1))
x = torch.randn(5,3).cuda()
y = torch.rand(5,3).cuda()
print(x+y)