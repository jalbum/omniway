# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
import csv

#NUM,BOSS,EVENT,SCALE,AGE,GENDER,ANSWER(FOOD)

BOSS = {1:'본부장',2:'사업부장',3:'실장',4:'팀장',5:'그룹장',6:'무'}
EVENT = {1:'망년신년',2:'승진', 3:'신입전입',4:'환송',5:'정기',6:'번개'}
SCALE = {1:'대_10명이상',2:'중_6~9명', 3:'소_0~5명'}
AGE = {1:'고)과차부장50%이상',2:'중)과차부장50%미만', 3:'저)사원대리50%이상'}
GENDER = {1:'여성',2:'여성과남성', 3:'남성'}
SEASON={1:'봄',2:'초여름',3:'더운여름',4:'가을',5:'초겨울',6:'추운겨울'}
WEATHER ={1:'맑음',2:'흐림',3:'비',4:'눈'}

i=0
for BO_K in BOSS.keys():
    for EV_K in EVENT.keys():
        for SC_K in SCALE.keys():
            for AG_K in AGE.keys():
               for GE_K in GENDER.keys():
                  for SE_K in SEASON.keys():
                      for WE_K in WEATHER.keys():
                          print(BO_K, ",", EV_K, ",", SC_K, ",", AG_K, ",", SE_K,",", GE_K,",", WE_K)
                          i +=1

print(i)

