'''
@Description:
@Author: HuangQinJian
@Date: 2019-02-17 15:39:19
@LastEditTime: 2019-03-17 18:19:51
@LastEditors: HuangQinJian
'''
import json
import os

import numpy as np
import pandas as pd

label_path = 'CSV/data/jinnan2_round1_train_20190305/train_no_poly.json'
restrict_img_path = 'CSV/data/jinnan2_round1_train_20190305/restricted/'
restrict_rele_path = 'data/jinnan2_round1_train_20190305/restricted/'


def restrict_image_info(label_path):

    with open(label_path, 'r') as load_f:
        load_dict = json.load(load_f)
        image_collect = load_dict['images']
        image_num = len(image_collect)
        anno_collect = load_dict['annotations']
        anno_num = len(anno_collect)

        img_path_list = []
        x1_list = []
        y1_list = []
        x2_list = []
        y2_list = []
        category_list = []

        mapper = {0: 'tieke', 1: 'heiding',
                  2: 'daoju', 3: 'dian', 4: 'jiandao'}

        for i in range(image_num):
            img = image_collect[i]
            img_name = img['file_name']
            img_id = img['id']
            img_height = img['height']
            img_width = img['width']

            for j in range(anno_num):
                if anno_collect[j]['image_id'] == img_id:
                    bbox = anno_collect[j]['bbox']
                    img_path_list.append(restrict_rele_path+img_name)
                    x1_list.append(int(np.rint(bbox[0])))
                    y1_list.append(int(np.rint(bbox[1])))
                    x2_list.append(
                        int(np.rint(bbox[0] + bbox[2])))
                    y2_list.append(
                        int(np.rint((bbox[1]+bbox[3]))))
                    category_list.append(anno_collect[j]['category_id']-1)

            anno = pd.DataFrame()
            anno['img_path'] = img_path_list
            anno['x1'] = x1_list
            anno['y1'] = y1_list
            anno['x2'] = x2_list
            anno['y2'] = y2_list
            anno['class'] = category_list
            anno['class'] = anno['class'].map(mapper)

        anno.to_csv('CSV/annotations.csv', index=None, header=None)


if __name__ == "__main__":
    restrict_image_info(label_path)
