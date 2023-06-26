import os
import numpy as np
import torch
import cv2

hm = cv2.imread('/home/marq/Desktop/MOT/vis/TraDeS/img.jpg')
img = cv2.imread('/home/marq/Desktop/MOT/vis/TraDeS/000001.jpg')

hm = cv2.resize(hm,(img.shape[1], img.shape[0]))
zeros = hm * 0
hm = np.where(hm>130, hm-100, zeros)
img_with_mask = hm  + img
cv2.imwrite('/home/marq/Desktop/MOT/vis/TraDeS/img_mask.jpg', img_with_mask)