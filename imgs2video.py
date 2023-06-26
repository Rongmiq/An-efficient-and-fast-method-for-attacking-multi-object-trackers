import cv2
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from data_utils import pre_process_normalize, post_process_normalize, normalize
# from pix2pix.GAN_utils import *
# from pix2pix.attack_utils import attack_imgs, get_noise

def imgs2video(img_dir, vedio_dir, video_name='', fps=20, h=1920, w=1080):

    # video_writer = cv2.VideoWriter(vedio_dir+ "/Video.avi", cv2.VideoWriter_fourcc('I', '4', '2', '0'), fps, (h, w))
    # video_writer = cv2.VideoWriter(vedio_dir + '/' + video_name + '.avi', cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'), fps, (h, w))
    video_writer = cv2.VideoWriter(vedio_dir + '/' + video_name + '.mp4', cv2.VideoWriter_fourcc(*"mp4v"), fps, (h, w))
    print(vedio_dir + '/' + video_name + '.mp4')
    images = os.listdir(img_dir)
    images = np.unique(images)
    fram = 0
    for i in images:
        if 0 <= fram < 2990:
            img_path = img_dir + '/' + i
            print(img_path)
            image = cv2.imread(img_path)
            # bg = cv2.imread('/home/marq/Desktop/MOT/test_videos/bg.jpg')
            video_writer.write(image)
        fram+=1
    video_writer.release()
    print('Total {} frames!'.format(fram))

def video2imgs(videoPath, videoPath1, videoPath2, videoPath3,videoPath4, videoPath5, vedio_dir):
    print(videoPath)
    cap = cv2.VideoCapture(videoPath)
    cap1 = cv2.VideoCapture(videoPath1)
    cap2 = cv2.VideoCapture(videoPath2)
    heat_clean_list = os.listdir(videoPath3)
    heat_blind_list = os.listdir(videoPath4)
    heat_blur_list = os.listdir(videoPath5)
    count = 500
    while(1):
        _, frame0 = cap.read()
        _, frame1 = cap1.read()
        _, frame2 = cap2.read()
        h_c = heat_clean_list[count-500]
        h_blind = heat_blind_list[count-500]
        h_blur = heat_blur_list[count-500]
        if frame0 is not None:
            # clean
            frame01 = cv2.resize(frame0, (444,250))
            # heatmaps
            frame02 = cv2.resize(cv2.imread(os.path.join(videoPath3,h_c)), (444,250))
            frame03 = cv2.resize(cv2.imread(os.path.join(videoPath4,h_blind)), (444,250))
            frame04 = cv2.resize(cv2.imread(os.path.join(videoPath5,h_blur)), (444,250))
            # blind and blur
            frame11 = cv2.resize(frame1, (900,506))
            frame12 = cv2.resize(frame2, (900,506))

            img0 = np.zeros([1080,1920,3])
            # blind
            img0[200:200+frame11.shape[0], 40:40+frame11.shape[1],:] = frame11
            # blur
            img0[200:200+frame12.shape[0], 900+40+40:980+frame12.shape[1],:] = frame12
            # clean
            img0[720:720+frame01.shape[0], 40:40+frame01.shape[1],:] = frame01
            # clean heatmaps
            img0[720:720+frame02.shape[0], 40+12+frame02.shape[1]: 52+ frame02.shape[1] * 2,:] = frame02
            # blind heatmaps
            img0[720:720+frame03.shape[0], 980:980+frame03.shape[1],:] = frame03
            # cblur heatmaps
            img0[720:720+frame04.shape[0], 980+frame04.shape[1]+12: 992+frame04.shape[1] * 2,:] = frame04
            if count <= 9:
                name = '0000'+str(count)
            elif count <= 99:
                name = '000' + str(count)
            elif count <= 999:
                name = '00' + str(count)
            elif count <= 9999:
                name = '0' + str(count)
            path = os.path.join('/home/marq/Desktop/MOT/test_videos/imgs/'+name+'.jpg')
            cv2.imwrite(path,img0)
            # assert p==0
            # imgs.append(img0)

            count += 1
            # video_writer.write(img0)
        else:
            # video_writer.release()
            cap.release()
            # print(len(imgs))
            # for i in range(len(imgs)):
            #     video_writer.write(imgs[i].astype(np.uint8))
            # video_writer.release()
            print(" %d "%(count-1))
            break


def add_noise_to_image(img_path, output_path, num_fram=[0,5000]):

    if img_path.endswith('jpg'):
        img = cv2.imread(img_path)
        img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).cuda()
        img_tensor = pre_process_normalize(img_tensor)
        adv_img_tensor = attack_imgs(img_tensor, GAN, bytetrack=False, fairmot=False)
        adv_img_tensor = post_process_normalize(adv_img_tensor)
        adv_img = adv_img_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        cv2.imwrite(output_path, adv_img)
    else:
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        image_list = os.listdir(img_path)
        assert len(image_list) != 0

        fram = 0
        for image in image_list:
            fram += 1
            if num_fram[1] >= fram >= num_fram[0]:
                img = cv2.imread(img_path+image)
                h, w, _ = img.shape
                img_t = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).cuda()
                img_t = pre_process_normalize(img_t)
                # img_resized = cv2.resize(img, (512,256))
                # img_resized = torch.from_numpy(img_resized.transpose(2,0,1)).unsqueeze(0).cuda()
                # img_resized = pre_process_normalize(img_resized)
                # noise = attack_imgs(img_t_r, GAN, bytetrack=False, fairmot=False, just_noise=True)
                noise = get_noise(img_t, GAN)

                noise = torch.nn.functional.interpolate(noise,size=(h,w),mode='bilinear')
                img = img_t + noise
                img = post_process_normalize(img)
                img = torch.clamp(img, min=0.0, max=255.0)
                img = img.squeeze(0).cpu().numpy().transpose(1, 2, 0)
                # adv_img = img + noise

                f = str(fram)

                # if len(str(fram)) == 1:
                #     f = '000' + str(fram)
                # elif len(str(fram)) == 2:
                #     f = '00' + str(fram)
                # elif len(str(fram)) == 3:
                #     f = '0' + str(fram)
                # adv_img_path = os.path.join(output_path+ f + '_adv.jpg')
                adv_img_path = os.path.join(output_path + image)

                cv2.imwrite(adv_img_path, img)
                print(adv_img_path)
        print("Process finished! %d images totally"%(num_fram[1]-num_fram[0]))

# img_dir = os.path.join('/home/marq/Desktop/datasets/MOT17/test/MOT17-14-DPM/img1/')
# video_dir = os.path.join('/home/marq/Desktop/datasets/MOT17/test/')
# imgs2video(img_dir, video_dir)

if __name__ == '__main__':
    # img_dirs = os.path.join('/home/marq/Desktop/datasets/MOT17/test/')
    # list = os.listdir(img_dirs)
    # new_list = []
    # for l in list:
    #     if "DPM" in l:
    #         new_list.append(l)
    # for l in new_list:
    #     print(l)
    #     img_dir = os.path.join(img_dirs, l+'/img1')
    #     print(img_dir)
    #     video_dir = os.path.join('/home/marq/Desktop/datasets/MOT17/test_videos')
    #     video_name = l + '_100'
    #     imgs2video(img_dir, video_dir, video_name)


    # video_path0 = os.path.join('/home/marq/Desktop/MOT/TraDeS/results/mot17_half_MOT17-14-DPM_100_clean.avi')
    # video_path1 = os.path.join('/home/marq/Desktop/MOT/TraDeS/results/mot17_half_MOT17-14-DPM_100_blind.avi')
    # video_path2 = os.path.join('/home/marq/Desktop/MOT/TraDeS/results/mot17_half_MOT17-14-DPM_100_blur.avi')
    # video_path3 = os.path.join('/home/marq/Desktop/MOT/test_videos/heatmaps_clean/')
    # video_path4 = os.path.join('/home/marq/Desktop/MOT/test_videos/heatmaps_blind/')
    # video_path5 = os.path.join('/home/marq/Desktop/MOT/test_videos/heatmaps_blur/')
    # output_path = os.path.join('/home/marq/Desktop/MOT/test_videos/')
    # video2imgs(video_path0,video_path1,video_path2,video_path3,video_path4, video_path5, output_path)
    #
    video_path = os.path.join('/home/marq/Desktop/MOT/ByteTrack/videos/7')
    output_path = os.path.join('/home/marq/Desktop/MOT/ByteTrack/videos/')
    imgs2video(video_path,output_path,video_name='7')


