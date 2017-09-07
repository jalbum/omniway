# output.csv를 읽어서 test데이타 수만큼 랜덤으로 추출하고 
# 추출한 것을 제외하고 train데이타를 만들어줌
# 다른 용도: 만약 test데이터수를 train데이터와 같게하면
#            test데이터는 train데이터를 무작위로 순서를 섞은 것과 같음
import sys, os
import numpy as np
import csv
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
dataset_dir = "dataset"

def read_csv_2(filename):
    list_n = [] # 추가
    list_x = []
    list_t = []
    f = open(filename, 'r')
    rdr = csv.reader(f)
    i = 0
    for line in rdr:
        if i == 0:
            list_head = line  # 헤드를 읽음
        else:
            line_int = list(map(int, line))  # csv를 읽으면 문자로 읽혀지므로 정수로 변환
            list_n.append(line_int[0]) # 추가: 첫번째 칼럼은 순번
            list_x.append(line_int[1:len(line_int) - 1]) # 첫번째 칼럼은 순번이므로 떼고 마지막도 답이므로 뗀다
            list_t.append(line_int[len(line) - 1])  # 마지막 칼럼은 답 태그
        i = 1
    f.close()
    my_h=np.array(list_head)
    my_n = np.array(list_n) #추가
    my_x = np.array(list_x)
    my_t = np.array(list_t)
    return my_h,my_n,my_x, my_t

def check_array(t_array,t_item):
    item_exist = False
    for i in range(0,t_array.size):
        if t_array[i]==t_item:
            item_exist = True
    return item_exist

def change_csv_str(s_array):
    sum_str=""
    for i in range(0, s_array.size):
        if i == 0:
            sum_str += str(s_array[i])
        else:
            sum_str += "," + str(s_array[i])
    return sum_str


# =================== Main =================
my_h,my_n,my_x, my_t = read_csv_2(dataset_dir + "/output_full.csv")
x_size = my_x.shape[0]
choice_size = 1000
mask = np.random.choice(x_size, choice_size)
x_choice = my_x[mask]
t_choice = my_t[mask]

#f1 = open(dataset_dir + '/output_train.csv', 'w', encoding='utf-8', newline='')  #python 3
f1 = open(dataset_dir + '/output_train.csv', 'w') #python 2
sum_str=change_csv_str(my_h)
f1.write(sum_str+"\n")

for i in range(0, x_size) :
    sum_str = str(my_n[i])+","+change_csv_str(my_x[i])
    sum_str += "," + str(my_t[i])+"\n"
    if check_array(mask,i)==False : #my_n 과 관계없이 순서만 관계있음
        f1.write(sum_str)

f1.close()

#f2 = open(dataset_dir + '/output_test.csv', 'w', encoding='utf-8', newline='') #python3
f2 = open(dataset_dir + '/output_test.csv', 'w') #python 2
sum_str=change_csv_str(my_h)
f2.write(sum_str+"\n")

for i in range(0,choice_size) :
    sum_str = str(mask[i])+","+change_csv_str(x_choice[i])
    sum_str += "," + str(t_choice[i])+"\n"
    f2.write(sum_str)

f2.close()

