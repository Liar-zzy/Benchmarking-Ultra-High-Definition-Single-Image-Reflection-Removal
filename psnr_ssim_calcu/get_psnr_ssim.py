import os
import os.path
import json
import argparse
from numpy import real
from psnr_ssim_calcu.psnr_ssim import calculate_psnr,calculate_ssim
import cv2
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--dataroot", default="", help="path to results roos")
parser.add_argument("--uhd", default="4k", help="4k or 8k")
parser.add_argument('--phase', type=str, default='test', help='train, test, etc')
parser.add_argument('--epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment. It decides where to store samples and models')

opt = parser.parse_args()



if __name__ == '__main__':
    

    dir = opt.dataroot
    # print(image_dir)
    json_path = os.path.join(dir,opt.uhd+'_'+opt.phase+'_'+opt.name+'_'+opt.epoch+'_results_list.json')

    with open(json_path,'r') as load_f:     
        input_list = json.load(load_f)
    load_f.close()

    results_name_list = []

    for index in input_list:
        results_name_list.append(index['test_output'])

    print("Cal %d results"%(len(results_name_list)))

    psnr_results = []
    ssim_results = []

    results_json_list = []
    index = 0
    for i in results_name_list:
        fakeT_path = os.path.join(dir, "images",i+'_fake_Ts_00.png')
        realT_path = os.path.join(dir, "images",i+'_real_T_00.png')
        fakeT = cv2.imread(fakeT_path, -1)
        # print(fakeT.shape)
        realT = cv2.imread(realT_path, -1)
        result_psnr = calculate_psnr(fakeT, realT, 0)
        psnr_results.append(result_psnr)

        result_ssim = calculate_ssim(fakeT, realT, 0)
        ssim_results.append(calculate_ssim(fakeT, realT, 0))
        print("processing %d/%d psnr/ssim "%(index,len(results_name_list)))
        results_json_list.append({'filename':i, 'psnr':result_psnr,'ssim':ssim_results})
        index+=1
    # 👆
    print("psnr : ")
    print(np.mean(psnr_results))
    # 👆
    print("ssim : ")
    print(np.mean(ssim_results))
    output_results = []
    output_results.append({'results':results_json_list,'psnr_result':np.mean(psnr_results),'ssim_result':np.mean(ssim_results)})

    json_path = os.path.join(dir,opt.uhd+'_'+opt.phase+'_'+opt.name+'_'+opt.epoch+'_psnr_ssim_results.json')

    with open(json_path,'w') as load_f:     
        json.dump(output_results, load_f)
    load_f.close()


    
