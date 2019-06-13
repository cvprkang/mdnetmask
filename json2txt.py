from __future__ import print_function
import json
import os

def load(path):
    f = open(path)
    result = json.load(f)
    res = result['res']
    return res

path = './result'
t_path = './result_n'
sq_name = os.listdir(path)

for i in range(len(sq_name)):
    json_path = os.path.join(path, sq_name[i], 'result.json')
    txt_name = sq_name[i] + '.txt'
    txt_path = os.path.join(t_path,txt_name)
    res = load(json_path)
    res = [str(res[i]) for i in range(len(res))]
    for j in range(len(res)):
        coor = res[j].strip('[]')
        coor = coor.replace(', ',' ')
        with open(txt_path, 'a') as file:
            print(coor, file = file)
'''
res = [str(t[i]) for i in range(len(t))]
for i in range(len(t)):
    coor = res[i].strip('[]')
    coor = coor.replace(', ',' ')
    with open('result.txt', 'a') as file:
        print(coor, file = file)
'''
