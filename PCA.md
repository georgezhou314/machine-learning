### Dimensionlity Reduction(降维)

### PCA(Principal components analysis)主成分分析方法

找一条直线，使得蓝色的线段之和最小，即投影误差最小。

![](https://github.com/georgezhou314/imageRepo/raw/master/ML/PCA1.png)

数据的预处理

1. 特征缩放

   确保每个特征的平均数是0

   X是N×1的矩阵
   $$
   Sigma=\frac{1}{m}\sum_{i=1}^{m}(x^i)(x^i)^T
   $$

 3.   降维算法
   ```matlab
   #Sigma是N×N的矩阵
   Sigma=(1/m)* x' *x;
   [U,S,V]=svd(Sigma)
   #产生的U矩阵，我们只用提取前K维即可，Ureduce是N×K维矩阵
   Ureduce=U(:,1:k);
   # Ureduce是N×K的矩阵,Ureduce'是K×N的矩阵，X是N×1的矩阵，所以Z是K×1的矩阵
   z=Ureduce' * x;
   ```

---

Choose K(number of principal components)   选择主成分

通常，选择k按照一下规则
$$
\frac{\frac{1}{m}\sum_{i=1}^{m}||x^i-x_{approx}^i||^2}{\frac{1}{m}\sum_{i=1}^m||x^i||^2}\le 0.01
$$
该规则说明了，算法保留了99%的方差

对于在MATLAB中可以用

其中S矩阵是，SVD算法返回的矩阵
$$
1-\frac{\sum_{i=1}^kS_{ii}}{\sum_{i=1}^nS_{ii}}\le0.01
$$

---

PCA算法将K维还原成N维

还原是近似还原。

```
# Ureduce是n×k维的矩阵，Z是K×1维的矩阵，所以Xapprox是N×1维的矩阵
Xapprox=Ureduce ×Z
```

---

PCA在监督学习中的使用

数据集合：(x1, y1), (x2, y2),..., (x(m),y(m))

x1,x2,..., x(m)假设是10000维的

经过(PCA)

压缩成1000维的z1, z2, ...,  z(m)

新的训练集合：(z1, y1), (z2, y2), ..., (z(m), y(m) )







