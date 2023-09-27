# Benchmarking-Ultra-High-Definition-Single-Image-Reflection-Removal

## Train

### 4k
```
python train_errnet.py --dataroot /nas_data/zhangzy/uhd_removal_dataset/4k/train --name 4kours \
--phase train --hyper --gpu_id 0 --uhd 4k --no-verbose --display_id -1
```

### 8k
```
python train_errnet.py --dataroot /nas_data/zhangzy/uhd_removal_dataset/8k/train --name 8kours \
--phase train --hyper --gpu_id 0 --uhd 8k --no-verbose --display_id -1
```

## Test

### 4k
```
python test_errnet.py --dataroot /nas_data/zhangzy/uhd_removal_dataset/4k/test/ --name 4kours \
--hyper --gpu_id -1 --uhd 4k -r --phase test --no-verbose --ctest 0
```

### 8k
```
python test_errnet.py --dataroot /nas_data/zhangzy/uhd_removal_dataset/8k/test/ --name 8kours \
--hyper --gpu_id -1 --uhd 8k -r --phase test --no-verbose --ctest 0
```

## Website
To explore more,please click https://liar-zzy.github.io/Benchmarking-UHD-SIRR/

## UHD4k-RR:4K Resolution Datasets

- 4k-train : https://pan.baidu.com/s/10kKgQH3m_aSEasmqK5oUDQ (wxao)
- 4k-test   : https://pan.baidu.com/s/1bEs5bEKGeELbWrXab685Qg (5a7x)

## UHD8K-RR:8K Resolution Datasets (Each 8K image is cutted into four 4K parts. You can concatenate to generate original 8K versions.)

- 8k-train : https://pan.baidu.com/s/1-vZ-FPh-gQtiE-jojwZX2A (f1r1)
- 8k-test   : https://pan.baidu.com/s/1hfQ4g3IsjSmkcaqo06tIng (mhzx)

## UHD4K-RR and UHD8K-RR Datasets

Google Drive To be released.

## Citation

If you think this work is useful for your research, please cite the folling paper.

```
@misc{zhang2023benchmarking,
      title={Benchmarking Ultra-High-Definition Image Reflection Removal}, 
      author={Zhenyuan Zhang and Zhenbo Song and Kaihao Zhang and Wenhan Luo and Zhaoxin Fan and Jianfeng Lu},
      year={2023},
      eprint={2308.00265},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

```
