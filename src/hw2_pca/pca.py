# -*- coding: utf-8 -*-
#!/usr/bin/env python

import os
import numpy as np
from numpy.linalg import svd
from pylab import *
import matplotlib.pyplot as plt

def zl_PCA(data):
    avg = np.average(data,0)
    means = data - avg
    #求Covariance矩阵
    tmp = matrix(means)*(matrix(means).T) 
    #求Covariance矩阵的特征值和特征向量
    D,V = np.linalg.eig(tmp)
    #取两个最大的特征向量
    E = V[:,0:2]
    M = matrix(E).T
    N = matrix(means)
    #对数据进行投影
    y = M * N
    return y

#读取数据并进行预处理
os.chdir("E:\ZL\mSpace\Python\csmath\src\hw2_pca")
filename = 'optdigits-orig.wdep'
inputData = []
X = []
#从文件中按行读取数据
for line in open(filename):
    if not line:
        break
    line = line.strip('\n')
    if len(line) < 5:
        number = int(line)
        if number == 3:#只保留形状为3的数据
            inputData.append(X)
        X=[]
    else:
       for str in line:
            X.append(int(str))
data = array(matrix(inputData).T)
#对数据进行zl_PCA处理，得到前两个向量
y = zl_PCA(data) 
#画图
# plt.plot(abs(y[:,0]),abs(y[:,1]))
plt.xlabel('X')
plt.ylabel('Y')
plt.title('PCA')
# plt.axis([0, 15, -10, 10])
plt.plot(abs(y[0,:]),abs(y[1,:]),'*')
# plt.setp(lines, color = 'r', linewidth = 2.0)
plt.show()