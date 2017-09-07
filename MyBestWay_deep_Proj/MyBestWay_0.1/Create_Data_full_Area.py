# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
import csv

#NUM,BOSS,EVENT,SCALE,AGE,GENDER,ANSWER(FOOD)

INPUT=[
        {1:'본부장',2:'사업부장',3:'실장',4:'팀장',5:'그룹장',6:'무'}, # 0 BOSS
        {1:'망년신년',2:'승진', 3:'신입전입',4:'환송',5:'정기',6:'비정기'}, #1 EVENT
        {1:'대_10명이상',2:'중_6~9명', 3:'소_0~5명'},#2 SCALE
        {1:'고)과차부장50%이상',2:'중)과차부장50%미만', 3:'저)사원대리50%이상'}, #3 AGE
        {1: '여성', 2: '여성과남성', 3: '남성'} , #4 GENDER
        {1:'봄',2:'여름',3:'가을',4:'겨울'}, #5 SEASON
        {1:'맑음',2:'흐림',3:'비',4:'눈'}, #6 WEATHER
        {1:'고', 2: '중', 3: '저'}  # 7 TEMP
        ]
OUTPUT=[ {1:'회사주변',2:'청계산',3:'양재',4:'사당',5:'강남',6:'의왕',7:'인덕원',8:'수원',9:'기타',11:"삭제"} ]# Area

def sel_out(k):
    o_y = 10
    if k[0] == 1  : # 본부장
        # 예외처리
        if k[1] != 1 and k[1] != 6:  # 망년도 아니고 비정기도 아니면
            o_y = 11  # 삭제
        if k[4] == 1:  # 여성이면
            if k[2] == 1: o_y = 11  # 인원 대는 없음
        if k[1] == 1:  # 망년이면
            if k[2] == 2 and k[2] == 3:  # 인원 중소는 없음
                o_y = 11

        # 조건처리
        if k[1]==1 : # 망년
           o_y = 2 #청계산
        if k[1]==6 : #비정기
            if k[2] == 2 and k[5]==4 and k[7]==3  : #인원 중, 겨울, 온도 저
                o_y = 1 #회사주변
            if k[2]== 1 and k[5]==2 and k[7]==1 :  # 인원 대, 여름, 온도고
                o_y = 3  # 양재
            if k[2]==2 and k[4]==1 : # 인원중 여성
                o_y = 5 #강남

    if k[0] == 2  or k[0]==3 : # 사업부장, 실장
        # 예외처리
        if k[5] == 2 and k[7]==1:  #  여름이고 온도고 면 한식
            o_y = 1  # 회사주변
        if k[4] == 1 : #  여성이면
            o_y = 5  # 강남

        # 조건 처리
        if k[1] == 6  : #비정기
            if k[2]==2 and k[5]==2 and k[4]== 2    :  # 인원중 여름, 여성남성
                o_y = 3 # 양재
            if k[2]==3 and k[4]==1 : #인원 소 , 여성
                o_y = 5  #강남
            if k[2]== 3 and k[4]== 3 : # 인원 소, 남성
                o_y = 1 # 회사주변
        if k[1] == 5 : #정기
            if k[2]== 2 and k[5]==4 and k[4]== 2    :  # 인원중 겨울, 여성남성
                o_y = 3 #양재
        if k[1] == 2 : # 승진
            if k[2] == 1 and k[5] == 4 and k[4] == 2:  # 인원대 겨울, 여성남성
                o_y = 2  # 청계산
        if k[1] == 3:  # 신입전입
            if k[2] == 2 and k[5] ==1 and k[4] == 2:  # 인원중, 봄, 여성남성
                o_y = 3  #양재
        if k[1] == 4:  # 환송
            if k[2] == 2 and  k[4] == 2:  # 인원중  여성남성
                o_y = 3 # 양재

    if k[0] == 4  : # 팀장
        # 예외처리
        if k[5] == 2 and k[7]==1:  #  여름이고 온도고 면 한식
            o_y = 1  # 회사주변
        if k[4] == 1 : #  여성이면
            o_y = 5  # 강남

        # 조건 처리
        if k[1] == 6  : #비정기
            if k[2]==2 and k[5]==2 and k[4]== 2    :  # 인원중 여름, 여성남성
                o_y = 3 # 양재
            if k[2]==3 and k[4]==1 : #인원 소 , 여성
                o_y = 5  #강남
            if k[2]== 3 and k[4]== 3 : # 인원 소, 남성
                o_y = 1 # 회자주변
        if k[1] == 5 : #정기
            if k[2]== 2 and k[5]==4 and k[4]== 2    :  # 인원중 겨울, 여성남성
                o_y = 3 #양재
        if k[1] == 2 : # 승진
            if k[2] == 1 and k[5] == 4 and k[4] == 2:  # 인원대 겨울, 여성남성
                o_y = 2  # 청계산
        if k[1] == 3:  # 신입전입
            if k[2] == 2 and k[5] ==1 and k[4] == 2:  # 인원중, 봄, 여성남성
                o_y = 3  # 양재
        if k[1] == 4:  # 환송
            if k[2] == 2 and  k[4] == 2:  # 인원중  여성남성
                o_y = 3  # 양재

    if k[5] == 4  or k[0]== 6 : # 그룹장, 無
        # 예외처리
        if  k[2] == 1 :  #인원 대 는 없음
            o_y = 11  # 삭제
        if k[1] != 6: #비정기만 있음
            o_y = 11  # 삭제
        if k[4] == 1: # 여성만 있는 경우는 없음
            o_y = 11  # 삭제
        # 조건처리
        if k[1] == 6:  # 비정기
           if k[2] == 3 and k[5] == 1 :  # 인원소 봄
                o_y =  5  #강남
           if k[2] == 3 and k[5] == 2:  # 인원소 여름
                o_y = 1 # 회사주변
           if k[2] == 3 and k[5] == 3:  # 인원소 가을
                o_y = 3  # 양재
           if k[2] == 3 and k[5] == 4:  # 인원소 가을
                o_y = 1  # 회사주변
           if k[2] == 2 and k[5] == 1:  # 인원중 봄
                o_y = 3  # 양재
           if k[2] == 2 and k[5] == 2:  # 인원중 여름
                o_y = 1  # 회사주변
           if k[2] == 2 and k[5] == 3:  # 인원중 가을
                o_y = 3  # 양재
           if k[2] == 2 and k[5] == 4:  # 인원중 겨울
                o_y = 2  # 청계산

    return o_y

#f = open('dataset/output_full.csv', 'w', encoding='utf-8',newline='') # python 3
f = open('dataset/output_full.csv', 'w') #python 2

#wr = csv.writer(f)#python 3
wr = csv.writer(f,lineterminator='\n')#python 2
k={}
cnt=0
n_cnt=0.00
# write Head
wr.writerow(["Area","B","C","D","E","F","G","H","I","J"])
for k[0] in INPUT[0].keys():#0
    for k[1] in INPUT[1].keys():#1
        for k[2] in INPUT[2].keys():#2
            for k[3] in INPUT[3].keys():#3
                for k[4] in INPUT[4].keys():  # 4
                    for k[5] in INPUT[5].keys():  # 5
                        for k[6] in INPUT[6].keys():  # 6
                            for k[7] in INPUT[7].keys():  # 6
                               y = sel_out(k)
                               if( y != 11 ) : # 11은 경우의 수에서 제외 (삭제)
                                   # print(k[0], ",",INPUT[0][k[0]],
                                   #       k[1], ",",INPUT[1][k[1]],
                                   #       k[2], ",",INPUT[2][k[2]],
                                   #       k[3], ",",INPUT[3][k[3]],
                                   #       k[4], ",",INPUT[4][k[4]],
                                   #       k[5], ",",INPUT[5][k[5]],
                                   #       k[6],",", INPUT[6][k[6]],
                                   #       k[7],",", INPUT[7][k[7]],
                                   #       y, OUTPUT[0][y]
                                   #       )
                                   if y!=10 :
                                      wr.writerow(
                                         [
                                             cnt,
                                             k[0], #INPUT[0][k[0]],
                                             k[1], #INPUT[1][k[1]],
                                             k[2], #INPUT[2][k[2]],
                                             k[3], #INPUT[3][k[3]],
                                             k[4], #INPUT[4][k[4]],
                                             k[5], #INPUT[5][k[5]],
                                             k[6], #INPUT[6][k[6]],
                                             k[7], #INPUT[7][k[7]],
                                             y,  #OUTPUT[0][y]
                                         ])
                                      cnt+=1
                                      if y==10 : n_cnt += 1 # 기타로 분리된 것 카운터

f.close()

print("전체 경우수 : " + str(cnt) + ",  매칭율 : " + str( ( (cnt-n_cnt)/cnt) ) )