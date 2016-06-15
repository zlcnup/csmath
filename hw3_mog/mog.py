# -*- coding: utf-8 -*-

#!/usr/bin/env python
from matplotlib import pyplot
from pylab import *
from numpy.linalg import det
import numpy as np
import numpy.matlib as ml
import random
from math import *

def init_params(centers,k):
    #使用kmeans算法设置EM算法的初始参数
    zl_pMiu = centers
    zl_pPi = zeros([1,k], dtype=float)
    zl_pSigma = zeros([len(X[0]), len(X[0]), k], dtype=float)
    #计算X到其他点的距离
    zl_dist = zl_distvec(X, centers)
    #分配X到最近的中心
    labels = zl_dist.argmin(axis=1)
    #重新计算每一个中心和参数
    for j in range(k):
        idx_j = (labels == j).nonzero()
        zl_pMiu[j] = X[idx_j].mean(axis=0)
        zl_pPi[0, j] = 1.0 * len(X[idx_j]) / nSamples
        zl_pSigma[:, :, j] = cov(mat(X[idx_j]).T)
    return zl_pMiu, zl_pPi, zl_pSigma
	
def zl_distvec(X, Y):
    n = len(X)
    m = len(Y)
    xx = ml.sum(X*X, axis=1)
    yy = ml.sum(Y*Y, axis=1)
    xy = ml.dot(X, Y.T)
    return tile(xx, (m, 1)).T+tile(yy, (n, 1)) - 2*xy

def calc_probability(k,zl_pMiu,zl_pSigma):
	#计算后验误差概率
    zl_Px = zeros([nSamples, k], dtype=float)
    for i in range(k):
        Xshift = mat(X - zl_pMiu[i, :])
        inv_pSigma = mat(zl_pSigma[:, :, i]).I
        coef = math.sqrt(2*3.14*det(mat(zl_pSigma[:, :, i])))
        for j in range(nSamples):
            tmp = (Xshift[j, :] * inv_pSigma * Xshift[j, :].T)
            zl_Px[j, i] = 1.0 / coef * math.exp(-0.5*tmp)
    return zl_Px

def data_generator(nSamples):
	#产生高斯混合模型
    mean = [15,15]
    cov = [[10,0],[0,100]]
    data = np.random.multivariate_normal(mean,cov,nSamples).T
    return data
    
def eStep(zl_Px, zl_pPi):
    #计算每个样本Xi由第K个函数产生的概率
    zl_pGamma =mat(array(zl_Px) * array(zl_pPi))
    zl_pGamma = zl_pGamma / zl_pGamma.sum(axis=1)
    return zl_pGamma

def mStep(zl_pGamma):
    zl_Nk = zl_pGamma.sum(axis=0)
    zl_pMiu = diagflat(1/zl_Nk) * zl_pGamma.T * mat(X) 
    zl_pSigma = zeros([len(X[0]), len(X[0]), k], dtype=float)
    for j in range(k):
        Xshift = mat(X) - zl_pMiu[j, :]
        for i in range(nSamples):
            zl_pSigmaK = Xshift[i, :].T * Xshift[i, :]
            zl_pSigmaK = zl_pSigmaK * zl_pGamma[i, j] / zl_Nk[0, j]
            zl_pSigma[:, :, j] = zl_pSigma[:, :, j] + zl_pSigmaK
    return zl_pGamma, zl_pMiu, zl_pSigma
	
def pylab_plot(X, labels, iter, k):
    colors = np.eye(k,k = 0)
    pyplot.plot(hold = False)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('MOG')
    plt.text(19,45, "Samples:%d  K:%d"%(len(X),k))
    
    labels = array(labels).ravel()
    data_colors = [colors[lbl] for lbl in labels]
    pyplot.scatter(X[:, 0], X[:, 1], c = data_colors, alpha = 0.5)
    pyplot.show()
    #pyplot.savefig('iter_%02d.png' % iter, format = 'png')

def MoG(X, k, threshold = 1e-10):
    N = len(X)
    labels = zeros(N, dtype = int)
    centers = array(random.sample(list(X), k))
    iter = 0
    zl_pMiu, zl_pPi, zl_pSigma = init_params(centers,k)
    Lprev = float('-10000')
    pre_esp = 100000
    while iter < 100:
        zl_Px = calc_probability(k,zl_pMiu,zl_pSigma)
        #EM算法的e-step
        zl_pGamma = eStep(zl_Px, zl_pPi)
        #EM算法的m-step
        zl_pGamma, zl_pMiu, zl_pSigma = mStep(zl_pGamma)
        labels = zl_pGamma.argmax(axis=1)
        #检查是否收敛
        L = sum(log(mat(zl_Px) * mat(zl_pPi).T))
        cur_esp = L-Lprev
        if cur_esp < threshold:
            break
        if cur_esp > pre_esp:
            break
        pre_esp = cur_esp
        Lprev = L
        iter += 1
    pylab_plot(X, labels, iter, k)

if __name__ == '__main__':
    #从控制台获取用户输入参数样本点数以及混合函数个数K
    print("Please Input the value of nSamples and  K")
    nSamples = input("Input nSamples: ")
    nSamples = int(nSamples)
    k = input("Input k (3 or 4): ")
    k = int(k)
    #生成高斯数据  
    samples = data_generator(nSamples)
    X = array(mat(samples).T)
    MoG(X, k)

