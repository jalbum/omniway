
# 파일명이나 기타 바꿀 것 없음(MyBest.py의 변경된 것 그대로 활용함)

# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
import pickle
from MyBest_dataset.MyBest import load_MyBest
from common.functions import sigmoid, softmax


def get_data():
    (x_train, t_train), (x_test, t_test) = load_MyBest(normalize=True,  one_hot_label=False)
    return x_test, t_test

def predict(network, x):
    W1, W2  = network.params['W1'], network.params['W2']
    b1, b2  = network.params['b1'], network.params['b2']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    y = softmax(a2)

    return y


x, t = get_data()
with open("dataset/Food/output_net.pkl", 'rb') as f:
    network = pickle.load(f)

accuracy_cnt = 0
for i in range(len(x)):
    y = predict(network, x[i])
    p= np.argmax(y) # 확률이 가장 높은 원소의 인덱스를 얻는다.
    if p == t[i]:
        accuracy_cnt += 1

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))
