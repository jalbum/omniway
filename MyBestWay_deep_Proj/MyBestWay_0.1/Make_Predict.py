
# 파일명이나 기타 바꿀 것 없음(MyBest.py의 변경된 것 그대로 활용함)

# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
import pickle
from Load_Data import load_MyBest
from common.functions import sigmoid, softmax

INPUT=[
        {1:'주체-본부장',2:'주체-사업부장',3:'주체-실장',4:'주체-팀장',5:'주체-그룹장',6:'주체-팀원'}, # 0 BOSS
        {1:'행사-망년신년',2:'행사-승진', 3:'행사-신입전입',4:'행사-환송',5:'행사-정기',6:'행사-비정기'}, #1 EVENT
        {1:'인원-대_10명이상',2:'인원-중_6~9명', 3:'인원-소_0~5명'},#2 SCALE
        {1:'연령-고)과차부장50%이상',2:'연령-중)과차부장50%미만', 3:'연령-저)사원대리50%이상'}, #3 AGE
        {1: '참가자-여성', 2: '참가자-여성과남성', 3: '참가자-남성'} , #4 GENDER
        {1:'계절-봄',2:'계절-여름',3:'계절-가을',4:'계절-겨울'}, #5 SEASON
        {1:'날씨-맑음',2:'날씨-흐림',3:'날씨-비',4:'날씨-눈'}, #6 WEATHER
        {1:'온도-고', 2: '온도-중', 3: '온도-저'}  # 7 TEMP
        ]
OUTPUT1=[ {1:'회사주변',2:'청계산',3:'양재',4:'사당',5:'강남',6:'의왕',7:'인덕원',8:'수원',9:'기타'} ]# Area
OUTPUT2=[ {1:'룸 의자식',2:'룸방바닥',3:'개방의자식',4:'개방방바닥'} ]# Type
OUTPUT3=[ {1:'한식',2:'중식',3:'일식',4:'퓨전/까페',5:'고기집(구이)',6:'횟집',7:'해산물',8:'뷔페',9:'국물요리',10:'기타',11:"삭제"} ]# FOOD

#==질문=======================================================================================
#        주체6, 이벤트6,  인원3,  연령3,  성별3,  계절4,  날씨4,  온도3
#Q =     [   4,       3,     1,      1,      2,      2,      1,      2 ] #보고서 예
#Q =      [   1,       1,     1,      1,      2,      3,      1,      2 ]
Q =      [   6,       3,     2,      3,     3,      2,     4,      1 ]
#=============================================================================================

def predict(network, x):
    W1, W2  = network.params['W1'], network.params['W2']
    b1, b2  = network.params['b1'], network.params['b2']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    y = softmax(a2)

    return y


print("Question--------------") #=======================================
for i in range(0,len(Q)):
    print(str(Q[i])+":" + INPUT[i][Q[i]])


print("Best Answer--------------") #=======================================

with open("dataset/Area/output_net.pkl", 'rb') as f:
    network1 = pickle.load(f)
y1 = predict(network1, Q)

p1= np.argmax(y1) # 확률이 가장 높은 원소의 인덱스를 얻는다.
print("장소 - " + str(p1) + ":"+str(OUTPUT1[0][p1]) )
f.close()

with open("dataset/Type/output_net.pkl", 'rb') as f:
    network2 = pickle.load(f)
y2 = predict(network2, Q)
p2= np.argmax(y2) # 확률이 가장 높은 원소의 인덱스를 얻는다.
print("형태 - " + str(p2) + ":"+ str(OUTPUT2[0][p2]))
f.close()


with open("dataset/Food/output_net.pkl", 'rb') as f:
    network3 = pickle.load(f)
y3 = predict(network3, Q)
p3= np.argmax(y3) # 확률이 가장 높은 원소의 인덱스를 얻는다.
print("음식 - " + str(p3) + ":"+ str(OUTPUT3[0][p3])+str(y3))
f.close()
