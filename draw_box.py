import cv2
import numpy
import os
from random import randint

colors = []
for i in range(200):
    color = randint(0, 0xFFFFFF)
    colorR = color % 0xFF
    color = color // 0xFF

    colorG = color % 0xFF
    color = color // 0xFF

    colorB = color
    colors.append((colorR, colorG, colorB))


def draw(img, bboxes):
    for box in bboxes:
        ids, ltx, lty, h, w = int(box[0]), int(box[1]), int(box[2]), int(box[3]), int(box[4])
        img = cv2.rectangle(img, (ltx, lty), (ltx+h, lty+w), colors[i], 2)
    img = cv2.putText(img, str(ids), (ltx, lty), 2, 2, colors[i], 2)
    cv2.imwrite('./img.jpg', img)

def load_bbox(txt_path, frame=1):
    bboxes = []
    with open(txt_path) as f:
        raw_data = f.readlines()
        for box in raw_data:
            bbox = []
            box = box.split(',')
            if int(box[0]) == frame:
                bbox.append(int(float(box[1])))
                bbox.append(int(float(box[2])))
                bbox.append(int(float(box[3])))
                bbox.append(int(float(box[4])))
                bbox.append(int(float(box[5])))

                bboxes.append(bbox)
    return bboxes


if __name__ == '__main__':
    bboxes = load_bbox(r'/home/marq/Desktop/datasets/MOT17/train/MOT17-04-DPM/det/det_train_half.txt')
    img_path = r'/home/marq/Desktop/datasets/MOT17/train/MOT17-04-DPM/img1/000001.jpg'
    img = cv2.imread(img_path)
    draw(img, bboxes)