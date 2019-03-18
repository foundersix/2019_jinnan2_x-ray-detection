#!/usr/bin/env python
# coding=UTF-8
'''
@Description: 
@Author: HuangQinJian
@LastEditors: HuangQinJian
@Date: 2019-03-07 10:45:48
@LastEditTime: 2019-03-16 12:05:59
'''
import json
import csv
import os
import pandas as pd

file_csv = 'submit.pkl'
file_json = 'submit.json'

r_csv = pd.read_pickle(file_csv)
w_json = open(file_json, 'w')

test_img_fold = 'keras_retinanet/CSV/data/jinnan2_round1_test_a_20190306/'
test_img_list = os.listdir(test_img_fold)
img_num = len(test_img_list)

results_list = []
# for i in range(2):
for i in range(img_num):
    dir_file = {}
    img_name = test_img_list[i]
    dir_file['filename'] = img_name
    one_img_list = []
    one_img_data = r_csv[r_csv.img_name == img_name]
    for j in range(len(one_img_data)):
        bbox = one_img_data.iloc[j]['bbox']
        xmin = int(bbox[0])
        ymin = int(bbox[1])
        xmax = int(bbox[2])
        ymax = int(bbox[3])
        label = int(one_img_data.iloc[j]['class'])
        score = float(one_img_data.iloc[j]['score'])
        if score >= 0.7:
            img_infs_dir = {}
            img_infs_dir['xmin'] = xmin
            img_infs_dir['xmax'] = xmax
            img_infs_dir['ymin'] = ymin
            img_infs_dir['ymax'] = ymax
            img_infs_dir['label'] = label+1
            img_infs_dir['confidence'] = round(score, 2)
            one_img_list.append(img_infs_dir)
    dir_file['rects'] = one_img_list
    results_list.append(dir_file)

result_dir = {}
result_dir['results'] = results_list
json.dump(result_dir, w_json, indent=4)
