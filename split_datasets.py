import os
import json

# mot_json_path = os.path.join('/home/marq/Desktop/datasets/MOT17/annotations/train_300_1_mot17.json')
# if not os.path.exists(mot_json_path):
#     os.mknod(mot_json_path)
#
# mot_dataset_path = os.path.join('/home/marq/Desktop/datasets/MOT17/train')
# video_list = os.listdir(mot_dataset_path)
# assert len(video_list) > 1
# video_new_list = []
# for i in range(len(video_list)):
#     video_new_list.append(video_list[i][0:8])
# video_new_list = list(set(video_new_list))
#
# video = {}
# images = []
# sub_video = {}
# ids = 0
# for i in range(len(video_new_list)):
#     for j in range(1,301):
#         if len(str(j)) == 1:
#             image_name = '00000' + str(j) + '.jpg'
#         elif len(str(j)) == 2:
#             image_name = '0000' + str(j) + '.jpg'
#         elif len(str(j)) == 3:
#             image_name = '000' + str(j) + '.jpg'
#         ids += 1
#         sub_video = {}
#         sub_video['file_name'] = video_new_list[i] + '-DPM/img1/' + image_name
#         sub_video['id'] = ids
#         images.append(sub_video)
#         # if j < 8:
#         #     image_name_add = '00000' + str(j+2) + '.jpg'
#         # elif 7 < j < 98:
#         #     image_name_add = '0000' + str(j+2) + '.jpg'
#         # else:
#         #     image_name_add = '000' + str(j+2) + '.jpg'
#         # ids += 1
#         # sub_video = {}
#         # sub_video['file_name'] = video_new_list[i] + '-DPM/img1/' + image_name_add
#         # sub_video['id'] = ids
#         # images.append(sub_video)
# video['images'] = images
#
# js = json.dumps(video)
# file = open(mot_json_path, 'w')
# file.write(js)
# file.close()


mot_json_path = os.path.join('/home/marq/Desktop/datasets/MOT20/annotations/train_mot20.json')
if not os.path.exists(mot_json_path):
    os.mknod(mot_json_path)

mot_dataset_path = os.path.join('/home/marq/Desktop/datasets/MOT20/train')
video_list = os.listdir(mot_dataset_path)
assert len(video_list) > 1
video_new_list = ['MOT20-01','MOT20-02','MOT20-03','MOT20-05']
print(video_new_list)
l = [420, 2784, 2400, 3312]
video = {}
images = []
sub_video = {}
ids = 0
for i in range(len(video_new_list)):
    for j in range(1,int(l[i]/2-2)):
        if len(str(j)) == 1:
            image_name = '00000' + str(j) + '.jpg'
        elif len(str(j)) == 2:
            image_name = '0000' + str(j) + '.jpg'
        elif len(str(j)) == 3:
            image_name = '000' + str(j) + '.jpg'
        elif len(str(j)) == 4:
            image_name = '00' + str(j) + '.jpg'
        ids += 1
        sub_video = {}
        sub_video['file_name'] = video_new_list[i] + '/img1/' + image_name
        sub_video['id'] = ids
        images.append(sub_video)
        if j < 8:
            image_name_add = '00000' + str(j+2) + '.jpg'
        elif 7 < j < 98:
            image_name_add = '0000' + str(j+2) + '.jpg'
        elif 97 < j < 998:
            image_name_add = '000' + str(j+2) + '.jpg'
        else:
            image_name_add = '00' + str(j+2) + '.jpg'
        ids += 1
        sub_video = {}
        sub_video['file_name'] = video_new_list[i] + '/img1/' + image_name_add
        sub_video['id'] = ids
        images.append(sub_video)
video['images'] = images

js = json.dumps(video)
file = open(mot_json_path, 'w')
file.write(js)
file.close()

