#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt


def curve_fitting(degree, num):
    x = np.linspace(0, 1)
    plt.plot(x, np.sin(2 * np.pi * x), 'g', linewidth=2, label='sin(2πx)')
    gaussian = np.random.normal(0, 0.1, num)
    rx = np.linspace(0, 1, num)
    ry = np.sin(2 * np.pi * rx) + gaussian
    plt.scatter(rx, ry, edgecolors='b', marker='>', linewidths=3, label='sample with noise')
    rz = np.polyfit(rx, ry, degree)
    rt = 0
    i = 0
    while i < rz.__len__():
        rt += rz[i] * x ** (rz.__len__() - 1 - i)
        i += 1
    plt.plot(x, rt, 'r', linewidth=2, label='curve fitting')
    plt.title('Polynomial Curve Fitting(degree=' + str(degree) + ', num=' + str(num) + ')')
    plt.xlabel('x')
    plt.ylabel('value')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    curve_fitting(3, 10)
    curve_fitting(9, 10)
    curve_fitting(9, 15)
    curve_fitting(9, 100)
