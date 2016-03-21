#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt


def main(degree, num):
    x = np.linspace(0, 1)
    plt.plot(x, np.sin(2 * np.pi * x), 'g', linewidth=2)
    gaussian = np.random.normal(0, 0.1, num)
    rx = np.linspace(0, 1, num)
    ry = np.sin(2 * np.pi * rx) + gaussian
    plt.scatter(rx, ry, edgecolors='b', marker='o', linewidths=3)
    rz = np.polyfit(rx, ry, degree)
    rt = 0
    i = 0
    while i < rz.__len__():
        rt += rz[i] * x ** (rz.__len__() - 1 - i)
        i += 1
    plt.plot(x, rt, 'r', linewidth=2)
    plt.title('Polynomial Curve Fitting(degree=' + str(degree) + ', num=' + str(num) + ')')
    plt.xlabel('x')
    plt.ylabel('value')
    plt.show()

if __name__ == '__main__':
    main(3, 10)
    main(9, 10)
    main(9, 15)
    main(9, 100)
