# train:
# 4k
python train_errnet.py --dataroot /nas_data/zhangzy/uhd_removal_dataset/4k/train --name 4kours \
--phase train --hyper --gpu_id 0 --uhd 4k --no-verbose --display_id -1

# 8k
python train_errnet.py --dataroot /nas_data/zhangzy/uhd_removal_dataset/8k/train --name 8kours \
--phase train --hyper --gpu_id 0 --uhd 8k --no-verbose --display_id -1


# test:
# 4k
python test_errnet.py --dataroot /nas_data/zhangzy/uhd_removal_dataset/4k/test_1050/ --name 4kours \
--hyper --gpu_id -1 --uhd 4k -r --phase test --no-verbose --ctest 0

# 8k
python test_errnet.py --dataroot /nas_data/zhangzy/uhd_removal_dataset/8k/test/ --name 8kours \
--hyper --gpu_id -1 --uhd 8k -r --phase test --no-verbose --ctest 0