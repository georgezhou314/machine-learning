---
title: 关于Normal Equation
date: 2019-10-23 14:37
tags:
- Machine Learning
---

已知数据集，x表示自变量，price(y)表示因变量

| x0   | Size（x1） | Number of bedrooms (x2) | Number of floors (x3) | Age of home (x4) | Price (y) |
| ---- | ---------- | ----------------------- | --------------------- | ---------------- | --------- |
| 1    | 2194       | 5                       | 1                     | 45               | 460       |
| 1    | 1416       | 3                       | 2                     | 40               | 232       |
| 1    | 1534       | 3                       | 2                     | 30               | 315       |
| 1    | 852        | 2                       | 1                     | 36               | 178       |

<!--more-->
由此，抽出X矩阵，Y矩阵


$$
X={\left[\begin{array}{c}
1 & 2194 & 5 & 1 &45 \\
1 & 1416 & 3 & 2 &42 \\
1 & 1534 & 3 & 2 &30 \\
1 & 852 & 2 & 1 &36 \\
\end{array}\right]}
$$

$$
y={\left[\begin{array}{c}
460\\
232\\
315\\
178
\end{array}\right]}
$$

#### 现在需要求出系数矩阵 θ

**公式推导**
$$
X是一个M\times(N+1)的矩阵，Y是M\times1矩阵，M是数据集的行数\\
\because X\times \theta =Y\\
\therefore X^{T}\times X\times \theta=X^T\times Y\\
经过化简： \theta=(X^T\times X)^{-1}\times X^T \times Y
$$


**Matlab中描述**

X'的英文是：X primes

```matlab
pinv(X'*X)*X'*y
```

