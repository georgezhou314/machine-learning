### 批量归一化，BatchNorm1d
批量归一化，第一个参数是维度，第二个参数是eps，加在分母上的，防止分母为1
批量归一化，具体参数：https://tangshusen.me/Dive-into-DL-PyTorch/#/chapter05_CNN/5.10_batch-norm
### LeakyReLU
LeakyReLU与ReLU在x<0的时候，处理方法不同
f(x)=αx，(x<0); f(x)=x,(x>0)

## Loss Function中
torch.nn.BCELoss是二分类交叉熵

### 规则化
Normalize(0.5,0.5)是将[0,1]的图片转换为[mean=0.5,std=0.5],转换完成范围为[-1,+1]
