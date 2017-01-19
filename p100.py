# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pylab as plt

def numerical_diff(f, x):
    h = 1e-4
    return (f(x+h) - f(x-h)) / (2*h)

def function_1(x):
    return 0.01 * x ** 2 + 0.1 * x

def tangent_line(f, x):
    # 傾き(f(a))
    d = numerical_diff(f, x)
    print("d"+str(d))
    # y-f(a) = f(a)(x-a)
    # y = f(a)(x-a) + f(a)
    #f(x)はx=5のときのy座標
    y = f(x) - d*x
    print("y" + str(y))
    return lambda t: d*t + y

x = np.arange(0.0, 20.0, 0.1)
y = function_1(x)

print(y)

plt.xlabel('x')
plt.ylabel('f(x)')

tf = tangent_line(function_1, 5)
print(tf)
y2 = tf(x)

plt.plot(x, y)
plt.plot(x, y2)
plt.show()
