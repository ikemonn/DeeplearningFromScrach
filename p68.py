# -*- coding: utf-8 -*-

import numpy as np
import pickle

import sys
resource_path = "/Users/daiki_matsui/Desktop/tmp/DeepLearningFromScratch/deep-learning-from-scratch"
sys.path.append(resource_path)
from dataset.mnist import load_mnist
from PIL import Image

def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()


def get_data():
    (x_train, t_train), (x_test, t_test) = \
        load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test

def init_network():
    with open(resource_path+"/ch03/sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network

def predict(network, x):
    W1, W2, W3 = network['W1'],network['W2'], network['W3']
    b1, b2, b3 = network['b1'],network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y

def mean_squared_error(y, t):
    return 0.5 * np.sum((y-t)**2)

def cronss_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t* np.log(y+delta))

def numerical_diff(f, x):
    h = 1e-4
    return (f(x+h) - f(x-h)) / (2*h)

def function_1(x):
    return 0.01 * x ** 2 + 0.1 * x




# x, t = get_data()
# network = init_network()
#
# batch_size = 100
# accuracy_cnt = 0
#
# for i in range(0, len(x), batch_size):
#     x_batch = x[i:i+batch_size]
#     y_batch = predict(network, x_batch)
#     p = np.argmax(y_batch, axis=1)
#     accuracy_cnt += np.sum(p == t[i:i+batch_size])
#
# print("Accuracy:" + str(float(accuracy_cnt/len(x))))