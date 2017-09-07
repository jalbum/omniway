# 하이퍼파리미터 바꾸어 줄 것

# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np

import matplotlib.pyplot as plt
from Load_Data import load_MyBest,dump_MyBest, read_input_size_csv
from two_layer_net import TwoLayerNet

# By Han : read_input_size_csv() 추가
# csv의 ,로 분리된 데이터수에서 번호부분과 답부분 2개를 빼면 input_sieze가 됨
# read_input_size_csv()는 미리 train데이터의 헤드를 읽어서 계산함
network = TwoLayerNet(input_size=read_input_size_csv(), hidden_size=50, output_size=11)  #network은 객체임

# 데이터 읽기
(x_train, t_train), (x_test, t_test) = load_MyBest(normalize=True, one_hot_label=True)
train_size = x_train.shape[0] #

# 하이퍼파라미터 : 장소 Area
# iters_num = 7000  # 반복 횟수를 적절히 설정한다.
# batch_size = 50  # 미니배치 크기
# learning_rate = 0.2

# 하이퍼파라미터 : 형태 Type
# iters_num = 6000  # 반복 횟수를 적절히 설정한다.
# batch_size = 200  # 미니배치 크기
# learning_rate = 0.3

# 하이퍼파라미터 한병혁  Food
iters_num = 10000  # 반복 횟수를 적절히 설정한다.
batch_size = 200  # 미니배치 크기
learning_rate = 0.3

train_loss_list = []
train_acc_list = []
test_acc_list = []

# 1에폭당 반복 수 : 전체 데이터수를 해당 배치만큼 처리할때 전체데이터를 모두 소진해 버리는 만큼의 회수
#                   전체 데이터에서 배치덩어리가 몇개인가? 로 생각할 수 있음
#                   아래 For문에서는 훈련횟수가 배치덩이리갯수만큼 될때마다(1에폭마다) 평가를 한번씩 해 본다
iter_per_epoch = max(train_size / batch_size, 1) # "훈련데이터수/배치수"가 1보다 작으면 1을 택하도록되어 있음

cnt_epoch=0
for i in range(iters_num):#1000
    # 미니배치 획득
    batch_mask = np.random.choice(train_size, batch_size) #
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 기울기 계산
    #grad = network.numerical_gradient(x_batch, t_batch)
    grad = network.gradient(x_batch, t_batch) #순전파/오차역전파

    # 매개변수 갱신 : 매 횟수마다 갱신해 나간다 
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]


    # 학습 경과 기록 = loss에서는 갱신된 매개변수로 predict(추론)을 하고 그때의 정확도를 측정해 본다
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    # 1에폭당 정확도 계산
    if int(i % iter_per_epoch)== 0:
        cnt_epoch += 1
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(str(cnt_epoch)+": train acc, test acc | " + str(round(train_acc,3)) + ", " + str(round(test_acc,3)))

# 매개변수 저장
dump_MyBest(network)

# 그래프 그리기
markers = {'train': 'o', 'test': 's'}
x = np.arange(len(train_acc_list))
plt.plot(x, train_acc_list, label='train acc')
plt.plot(x, test_acc_list, label='test acc', linestyle='--')
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()