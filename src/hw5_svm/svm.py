# -*- coding: utf-8 -*-

#!/usr/bin/env python
import numpy as np
from numpy import linalg
import cvxopt
import cvxopt.solvers
import pylab as pl

def gaussian_kernel(x, y, sigma=5.0):
    return np.exp(-linalg.norm(x-y)**2 / (2 * (sigma ** 2)))

def data_generator(nSamples):
    zl_mean1 = [-1, 2]
    zl_mean2 = [1, 1]
    zl_mean3 = [-2, 2]
    zl_mean4 = [-4, 4]
    zl_cov = [[1.0,0.8], [0.8,1.0]]
    X1 = np.random.multivariate_normal(zl_mean1, zl_cov, nSamples)
    X1 = np.vstack((X1, np.random.multivariate_normal(zl_mean3, zl_cov, nSamples)))
    Y1 = np.ones(len(X1))
    X2 = np.random.multivariate_normal(zl_mean2, zl_cov, nSamples)
    X2 = np.vstack((X2, np.random.multivariate_normal(zl_mean4, zl_cov, nSamples)))
    Y2 = np.ones(len(X2)) * -1
    myfile = open("test.txt", 'w')
    print(myfile, X1)
    return X1, Y1, X2, Y2

def zl_splitData(X1, Y1, X2, Y2):
	#分为训练集及测试集
    splitLine = int(len(X1)*0.9)
    X_train = np.vstack((X1[:splitLine],X2[:splitLine]))
    Y_train = np.hstack((Y1[:splitLine],Y2[:splitLine]))   
    X_test = np.vstack((X1[splitLine:],X2[splitLine:]))
    Y_test = np.hstack((Y1[splitLine:],Y2[splitLine:]))
    return X_train, Y_train, X_test, Y_test

def zl_pylab_plot_contour(X1_train, X2_train, self_a, self_sv_y, self_sv):
    pl.xlabel('X')
    pl.ylabel('Y')
    pl.title('SVM')
    pl.plot(X1_train[:,0], X1_train[:,1], "ro")
    pl.plot(X2_train[:,0], X2_train[:,1], "yo")
    pl.scatter(self_sv[:,0], self_sv[:,1], s=100, c="g")
    X1, X2 = np.meshgrid(np.linspace(-6,6,50), np.linspace(-6,6,50))
    X = np.array([[x1, x2] for x1, x2 in zip(np.ravel(X1), np.ravel(X2))])
    Z = zl_project(X ,self_a, self_sv_y, self_sv).reshape(X1.shape)
    pl.contour(X1, X2, Z, [0.0], colors='k', linewidths=1, origin='lower')
    pl.contour(X1, X2, Z + 2, [0.0], colors='grey', linewidths=1, origin='lower')
    pl.contour(X1, X2, Z - 2, [0.0], colors='grey', linewidths=1, origin='lower')
    pl.axis("tight") #使坐标系的最大值和最小值和数据范围一致
    pl.show()

def zl_svm_fit(X, Y):
    n_samples, n_features = X.shape
    K = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            K[i,j] = gaussian_kernel(X[i], X[j])
    zl_P = cvxopt.matrix(np.outer(Y,Y) * K)
    zl_q = cvxopt.matrix(np.ones(n_samples) * -1)
    zl_A = cvxopt.matrix(Y, (1,n_samples))
    zl_b = cvxopt.matrix(0.0)
    zl_G = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
    zl_h = cvxopt.matrix(np.zeros(n_samples))
    # solve QP problem
    solution = cvxopt.solvers.qp(zl_P, zl_q, zl_G, zl_h, zl_A, zl_b)
    # Lagrange multipliers
    a = np.ravel(solution['x'])
    # Support vectors have non zero lagrange multipliers
    sv = a > 1e-5
    ind = np.arange(len(a))[sv]
    self_a = a[sv]
    self_sv = X[sv]
    self_sv_y = Y[sv]
    print("%d support vectors out of %d points" % (len(self_a), n_samples))
    # Intercept
    self_b = 0
    for n in range(len(self_a)):
        self_b += self_sv_y[n]
        self_b -= np.sum(self_a * self_sv_y * K[ind[n],sv])
    self_b /= len(self_a)
    return self_a,self_sv,self_sv_y,self_b

def zl_project(X,self_a,self_sv_y,self_sv):
    y_predict = np.zeros(len(X))
    for i in range(len(X)):
        s = 0
        for a, sv_y, sv in zip(self_a, self_sv_y, self_sv):
            s += a * sv_y * gaussian_kernel(X[i], sv)
        y_predict[i] = s
    return (y_predict + self_b)

def zl_predict(X,self_a,self_sv_y,self_sv):
    return np.sign(zl_project(X,self_a,self_sv_y,self_sv))

if __name__ == "__main__":
    print("Please input nSamples of each center:")
    nSamples = input("Input nSamples:")
    nSamples = int(nSamples)
    X1, Y1, X2, Y2 = data_generator(nSamples)
    X_train, Y_train, X_test, Y_test = zl_splitData(X1, Y1, X2, Y2)
    self_a,self_sv,self_sv_y,self_b = zl_svm_fit(X_train, Y_train)
    Y_predict = zl_predict(X_test,self_a,self_sv_y,self_sv)
    correct = np.sum(Y_predict == Y_test)
    print("%d out of %d predictions correct" % (correct, len(Y_predict)))
    zl_pylab_plot_contour(X_train[Y_train==1], X_train[Y_train==-1], self_a, self_sv_y, self_sv)