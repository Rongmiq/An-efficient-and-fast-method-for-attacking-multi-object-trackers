import os

dir = '/home/marq/Desktop/datasets/MOT16/train'
file_list = os.listdir(dir)
print(file_list)
for f in file_list:
    txt_path = os.path.join(dir, f, 'gt/gt.txt')
    new_txt_path = os.path.join(dir, f, 'gt/{}.txt'.format(f))

    if os.path.exists(new_txt_path):
        os.rename(new_txt_path, txt_path)

    # if os.path.exists(txt_path):
    #     os.rename(txt_path, new_txt_path)
