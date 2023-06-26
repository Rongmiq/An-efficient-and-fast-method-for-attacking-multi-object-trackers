import os
import cv2
import torch
import numpy as  np

l2 = torch.nn.MSELoss()
def my_l2(adv_img, clean_img):
    B, C, W, H = adv_img.shape
    noise_2 = (adv_img - clean_img) ** 2
    noise_2[:,0,:,:] *= 0.8
    noise_2[:,1,:,:] *= 1
    noise_2[:,2,:,:] *= 1.2
    image_noise = torch.mean(noise_2)
    return image_noise

def linf(adv_img, clean_img):
        # [B,2,128*128]
        B, C, W, H = adv_img.shape
        image_noise = torch.abs(adv_img - clean_img).view(B,-1)
        max_value = torch.max(image_noise,dim=-1)
        l = torch.mean(max_value[0])
        return l

def pre_process_normalize(images):
    mean = np.array([0.40789654, 0.44719302, 0.47026115],
                     dtype=np.float32).reshape(1, 3, 1, 1)
    std  = np.array([0.28863828, 0.27408164, 0.27809835],
                     dtype=np.float32).reshape(1, 3, 1, 1)
    mean_tensor = torch.from_numpy(mean).cuda()
    std_tensor = torch.from_numpy(std).cuda()
    images_normalize = ((images / 255. - mean_tensor) / std_tensor).float()
    images_normalize = images_normalize.to()
    return images_normalize

def post_process_normalize(images):
    mean = np.array([0.40789654, 0.44719302, 0.47026115],
                     dtype=np.float32).reshape(1, 3, 1, 1)
    std  = np.array([0.28863828, 0.27408164, 0.27809835],
                     dtype=np.float32).reshape(1, 3, 1, 1)
    mean_tensor = torch.from_numpy(mean).cuda()
    std_tensor = torch.from_numpy(std).cuda()

    images_post_normalize = (images * std_tensor + mean_tensor) * 255.

    return images_post_normalize

if __name__ == '__main__':

    # adv_img = cv2.imread('/home/marq/Desktop/MOT/checkpoints/G_reid2_L2_500/web/images/epoch001_image_adv_vis.png')
    # clean_img = cv2.imread('/home/marq/Desktop/MOT/checkpoints/G_reid2_L2_500/web/images/epoch001_image_clean_vis.png')
    # #
    # adv_tensor = torch.from_numpy(adv_img.transpose(2,0,1)).cuda()
    # clean_tensor = torch.from_numpy(clean_img.transpose(2,0,1)).cuda()
    #
    # adv_tensor = pre_process_normalize(adv_tensor.unsqueeze(0))
    # clean_tensor = pre_process_normalize(clean_tensor.unsqueeze(0))
    #
    # # noise_tensor_post = post_process_normalize(adv_tensor) - post_process_normalize(clean_tensor)
    # noise_add = adv_tensor - clean_tensor
    # noise = torch.abs(torch.min(noise_add)) + noise_add
    # noise /= torch.max(noise)
    # noise *= 255
    # noise = noise.squeeze(0).cpu().numpy().transpose(1,2,0)
    # # noise[:,:,1:3] *= 0
    # cv2.imwrite('/home/marq/Desktop/MOT/vis/noise/noise.jpg',noise)
    # l2 = l2(adv_tensor, clean_tensor) * 500
    # linf = linf(adv_tensor, clean_tensor) * 10
    # my_l2 = my_l2(adv_tensor, clean_tensor) * 500
    # print(l2)
    # print(my_l2)
    # print(linf)
    # noise = cv2.imread('/home/marq/Desktop/MOT/vis/noise/noise.jpg')
    #
    # # r = noise[:,:,0]
    # # g = noise[:,:,1]
    # # b = noise[:,:,2]
    # # r_sum = np.sum(r)
    # # g_sum = np.sum(g)
    # # b_sum = np.sum(b)
    # # print(r_sum,g_sum,b_sum)
    #
    # g = 1.2
    # b = 1
    # r = 0.8
    #
    # noise[:,:,0] = noise[:,:,0] * g
    # noise[:,:,1] = noise[:,:,1] * b
    # noise[:,:,2] = noise[:,:,2] * r
    #
    # cv2.imwrite('/home/marq/Desktop/MOT/vis/noise/noise_new.jpg', noise)
    #
    # noise_1 = clean_img + noise
    # cv2.imwrite('/home/marq/Desktop/MOT/vis/noise/noise_1.jpg', noise_1)

    import os
    from matplotlib import pyplot as plt
    img_l = ['MOT17-02-DPM', 'MOT17-04-DPM', 'MOT17-05-DPM', 'MOT17-09-DPM', 'MOT17-10-DPM', 'MOT17-11-DPM',
             'MOT17-13-DPM']
    for i in range(len(img_l)):
        img_list = os.path.join('/home/marq/Desktop/datasets/MOT17/train/',img_l[i],'img1','000001.jpg')
        img = cv2.imread(img_list)
        if img.shape[0] != 1080:
            img = cv2.resize(img,(1920,1080))
        if i == 0:
            imgs = img
        else:
            imgs = np.concatenate((imgs, img), -1)
    imgs = imgs.reshape(-1)
    print(imgs.shape)
    num_bin = range(0, 255, 10)
    plt.hist(imgs, num_bin)
    plt.xticks(num_bin)
    plt.savefig('/home/marq/Desktop/MOT/vis/plt111.png')