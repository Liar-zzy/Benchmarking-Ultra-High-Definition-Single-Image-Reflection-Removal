import lpips
import os
from os.path import join
from glob import glob
import numpy as np
import torch
# torch限制cpu
torch.set_num_threads(5)
vgg_loss = lpips.LPIPS(net='vgg').cuda()

# mix_results_dir = 'results/4kours_168'
mix_results_dir = 'results/8kcrop_256_test'
# mix下的文件夹列表
mix_results_list = os.listdir(mix_results_dir)
# 遍历整个每个文件夹，下面的真值为m_gt.png，预测值为4k_raw.png
gt_results = []
pred_results = []
for mix_results in mix_results_list:
    if 'json' in mix_results :
        continue
    # 真值
    gt_results.append(join(mix_results_dir, mix_results, 'm_gt.png'))
    # 预测值
    # pred_results.append(join(mix_results_dir, mix_results, '4kcrop_256.png'))
    pred_results.append(join(mix_results_dir, mix_results, '8kcrop_256.png'))

print(len(gt_results), len(pred_results))

# 用于存放每一张图片的LPIPS值
lpips_list = []
for gt, pred in zip(gt_results, pred_results):
    # 读取图片并转化为tensor
    print(gt, pred)
    gt_img = lpips.im2tensor(lpips.load_image(gt)).cuda()
    pred_img = lpips.im2tensor(lpips.load_image(pred)).cuda()
    # 计算LPIPS值
    lpips_val = vgg_loss(gt_img, pred_img)
    lpips_list.append(lpips_val.item())

# 计算平均值
lpips_mean = np.mean(lpips_list)
print('LPIPS: ', lpips_mean)
