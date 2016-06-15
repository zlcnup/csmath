# -*- coding: utf-8 -*-

#!/usr/bin/enzl_v python
from pylab import *
from numpy import *
from math import *

def data_generator(N):
    #生成向量函数F：ai*exp(bi*x)的系数数组
    zl_mean = [3.4,4.5]
    zl_cozl_v = [[1,0],[0,10]]
    zl_coff = np.random.multivariate_normal(zl_mean,zl_cozl_v,N)
    #生成观测值向量y
    x = np.random.uniform(1, N, N)
    y = [zl_coff[i][0]*exp(-zl_coff[i][1]*x[i]) for i in range(N)]
    #生成初始值x0
    x0 = [x[i]+np.random.normal(0.0,0.3) for i in range(N)]
    return zl_coff, y, x0

def jacobian(zl_coff, x0, N):
    J=zeros((N,N),float)
    #计算第i个函数对X的第j个维度变量的偏导数
    for i in range(N):
        for j in range(N):
            #-abexp(-b*xi)
            J[i][j] = -(zl_coff[i][0]*zl_coff[i][1])*exp(-(zl_coff[i][1]*x0[j]))
    return J
    
def normG(g):
    absg = abs(g)
    Normg = absg.argmax()
    num = absg[Normg]
    return num
    
def zl_LM(zl_coff, y, x0, N, maxIter):
    zl_numIter = 0
    zl_v = 2
    zl_miu = 0.05 #阻尼系数
    x = x0
    zl_Threshold = 1e-5
    zl_preszl_fx = 100000
    while zl_numIter < maxIter:
        zl_numIter += 1
        #计算Jacobian矩阵
        J = jacobian(zl_coff, x, N)
        #计算Hessian矩阵，Ep以及g值
        A = dot(J.T,J)
        zl_fx = zeros((N,N),float)
        zl_fx = [zl_coff[i][0]*exp(-zl_coff[i][1]*x[i]) for i in range(N)]
        szl_fx = sum(array(zl_fx)*array(zl_fx))
        Ep = array(y) - array(zl_fx)
        g = array(dot(J.T,Ep))            
        H = A + zl_miu*np.eye(N)
        DTp = solve(H, g)
        x = x + DTp
        zl_fx2 = zeros(N,float)
        for j in range(N):
            zl_fx2[j] = zl_coff[j][0]*exp(-zl_coff[j][1])
        szl_fx2 = sum(array(zl_fx2)*array(zl_fx2))
        if abs(szl_fx - zl_preszl_fx) < zl_Threshold:
            print("The zl_vector x is: ")
            print(x)
            print("The sum is: ")
            print(szl_fx2)
            break 
        if szl_fx2 < (szl_fx+0.5*sum(array(g)*array(Ep))):
            zl_miu /= zl_v
        else :
            zl_miu *= 2
    if zl_numIter == maxIter:
        print("The zl_vector x0 is: ")
        print(x0)
        print("The zl_vector x is: ")
        print(x)
        print("The sum is: ")
        print(szl_fx2)

    
if __name__ == "__main__":
    #输入向量空间的长度N（在这里假设m=n）
    print("Please Input the dimension N of zl_vector space and the maxIter (the product of N and maxIter not be too large)")
    N = input("Input N (not be too large): ")
    N = int(N)
    maxIter = input("Input the max number of interation (larger than half of the N): ")
    maxIter = int(maxIter)
    zl_coff, y, x0 = data_generator(N)
    #zl_LM算法
    zl_LM(zl_coff, y, x0, N, maxIter)

