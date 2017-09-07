# 파일명 바꾸어 줄 것

# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
import csv
import pickle
# dataset_dir = os.path.dirname(os.path.abspath(__file__))
dataset_dir = "dataset"


# 장소 Area
# save_file_train = dataset_dir + "/Mybest_Area_train.csv"
# save_file_test = dataset_dir + "/Mybest_Area_test.csv"
# save_file_weight = dataset_dir + "/MyBest_Area_filter.pkl"

# 음식 Food
# save_file_train = dataset_dir + "/Mybest_Food_train.csv"
# save_file_test = dataset_dir + "/Mybest_Food_test.csv"
# save_file_weight = dataset_dir + "/MyBest_Food_filter.pkl"

# 형태 Type
# save_file_train = dataset_dir + "/Mybest_Type_train.csv"
# save_file_test = dataset_dir + "/Mybest_Type_test.csv"
# save_file_weight = dataset_dir + "/MyBest_Type_filter.pkl"


# 한병혁 테스트
save_file_train = dataset_dir + "/output_train.csv"
save_file_test = dataset_dir + "/output_test.csv"
save_file_weight = dataset_dir + "/output_net.pkl"

def _change_ont_hot_label(X):
    T = np.zeros((X.size, 11))
    for idx, row in enumerate(T):
        row[X[idx]] = 1
    return T

def dump_MyBest(network):
    # 매개변수 저장
    with open(save_file_weight, 'wb') as output:
        pickle.dump(network, output)
    return True

def read_input_size_csv():
    filename=save_file_train
    f = open(filename, 'r', encoding='utf-8')
    rdr = csv.reader(f)
    for line in rdr:
        input_size=len(line)-2
        break
    f.close()
    return  input_size

# csv를 읽기 위해 별도 만든 펑션
def read_csv(filename):
    list_x = []
    list_t = []
    f = open(filename, 'r', encoding='utf-8')
    rdr = csv.reader(f)
    i = 0
    for line in rdr:
        if i == 0:
            list_head = line  # 헤드를 읽음
        else:
            line_int = list(map(int, line)) # csv를 읽으면 문자로 읽혀지므로 정수로 변환
            list_x.append(line_int[1:len(line_int) - 1]) # 첫번째 칼럼은 순번이므로 떼고 마지막도 답이므로 뗀다
            list_t.append(line_int[len(line) - 1])  # 마지막 칼럼은 답 태그
        i = 1
    f.close()
    my_x = np.array(list_x)
    my_t = np.array(list_t)
    return my_x, my_t

def load_MyBest(normalize=False, one_hot_label=False):

    my_x_train , my_t_train = read_csv(save_file_train)
    my_x_test, my_t_test =  read_csv(save_file_test)

    if one_hot_label:
        my_t_train = _change_ont_hot_label(my_t_train)
        my_t_test = _change_ont_hot_label(my_t_test)

    if normalize:
        for norm in my_x_train:
            norm = norm.astype(np.float32)
            norm /= 10 # x의 값이 대체로 1~10값이므로
        for norm in my_t_train:
            norm = norm.astype(np.float32)
            norm /= 10 # x의 값이 대체로 1~10값이므로

    return (my_x_train, my_t_train), (my_x_test, my_t_test)

