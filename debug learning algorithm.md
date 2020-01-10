debug learning algorithm

**偏差：对于训练集而言的误差；高偏差，欠拟合导致**

**方差：对于测试集而言的误差；高方差：过拟合导致**

* 拿到更多训练数据 (修复高方差)
* try smaller sets of feature(修复高方差）
* try getting additional features(修复高偏差)
* try adding polynomial features(x1^2,x2^2,x1x2,etc)，修复高偏差
* try decreasing λ (修复高偏差，因为λ是高阶项惩罚系数，减小λ会导致高阶项的权重加大)
* try increasing λ(修复高方差)

### 神经网络的层数

* 隐藏层有较少的节点：计算量小，可能会导致欠拟合
* 隐藏层有多个节点：计算量大，可能会导致过拟合，不过可以用较大的λ来矫正

