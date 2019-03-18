#!/usr/bin/env python
# coding=UTF-8
'''
@Description: 
@Author: HuangQinJian
@LastEditors: HuangQinJian
@Date: 2019-03-08 19:04:59
@LastEditTime: 2019-03-08 20:49:56
'''

import os
import cv2
from imgaug import augmenters as iaa


if not os.path.exists('CSV/data/IMG_AUG/'):
    os.mkdir('CSV/data/IMG_AUG/')

img_path = 'CSV/data/jinnan2_round1_train_20190305/restricted/'
img_aug_path = "CSV/data/IMG_AUG/"

seq = iaa.Sometimes(p=0.5,
                    then_list=[iaa.GaussianBlur(
                        sigma=(1.0, 1.7)), iaa.AllChannelsCLAHE(clip_limit=5)],
                    else_list=[iaa.GammaContrast(gamma=3)],
                    name=None,
                    deterministic=False,
                    random_state=None)

imglist = []
img_names = os.listdir(img_path)
print(len(img_names))
i = 0
for img_name in img_names:
    # if i <= 20:
    #     img = cv2.imread(os.path.join(img_path, img_name))
    #     imglist.append(img)
    #     i += 1
    img = cv2.imread(os.path.join(img_path, img_name))
    imglist.append(img)
    i+=1

images_aug = seq.augment_images(imglist)
print(len(images_aug))
print(i)

for j in range(i):
    raw_img_name = img_names[j]
    aug_img_name = img_names[j].split('.')[0]+'_aug.jpg'
    # cv2.imwrite(os.path.join(img_aug_path, raw_img_name), imglist[j])
    cv2.imwrite(os.path.join(img_aug_path, aug_img_name), images_aug[j])

