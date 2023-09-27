import os.path
from os.path import join
from data.image_folder import make_dataset, make_dataset_uhd, make_dataset_uhd_crop
from data.transforms import Sobel, to_norm_tensor, to_tensor, ReflectionSythesis_1, ReflectionSythesis_2
from PIL import Image
import random
import torch
import math

import torchvision.transforms as transforms
import torchvision.transforms.functional as F

import util.util as util
import data.torchdata as torchdata


def __scale_width(img, target_width):
    ow, oh = img.size
    if (ow == target_width):
        return img
    w = target_width
    h = int(target_width * oh / ow)
    h = math.ceil(h / 2.) * 2  # round up to even
    return img.resize((w, h), Image.BICUBIC)

def __scale_height(img, target_height):
    ow, oh = img.size
    if (oh == target_height):
        return img
    h = target_height
    w = int(target_height * ow / oh)
    w = math.ceil(w / 2.) * 2  # round up to even
    return img.resize((w, h), Image.BICUBIC)


def paired_data_transforms(img_1, img_2, unaligned_transforms=False):
    def get_params(img, output_size):
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw
    
    # target_size = int(random.randint(224+10, 448) / 2.) * 2
    target_size = int(random.randint(224, 448) / 2.) * 2
    # target_size = int(random.randint(256, 480) / 2.) * 2
    ow, oh = img_1.size
    if ow >= oh:
        img_1 = __scale_height(img_1, target_size)
        img_2 = __scale_height(img_2, target_size)
    else:
        img_1 = __scale_width(img_1, target_size)
        img_2 = __scale_width(img_2, target_size)

    if random.random() < 0.5:
        img_1 = F.hflip(img_1)
        img_2 = F.hflip(img_2)

    i, j, h, w = get_params(img_1, (224,224))
    # i, j, h, w = get_params(img_1, (256,256))
    img_1 = F.crop(img_1, i, j, h, w)
    
    if unaligned_transforms:
        # print('random shift')
        i_shift = random.randint(-10, 10)
        j_shift = random.randint(-10, 10)
        i += i_shift
        j += j_shift

    img_2 = F.crop(img_2, i, j, h, w)
    
    return img_1,img_2

BaseDataset = torchdata.Dataset


class DataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size, shuffle, *args, **kwargs):
        super(DataLoader, self).__init__(dataset, batch_size, shuffle, *args, **kwargs)
        self.shuffle = shuffle

    # def reset(self):
        # if self.shuffle:
            # print('Reset Dataset...')
            # self.dataset.reset()


class CEILDataset(BaseDataset):
    def __init__(self, datadir, fns=None, size=None, enable_transforms=True, low_sigma=2, high_sigma=5, low_gamma=1.3, high_gamma=1.3):
        super(CEILDataset, self).__init__()
        self.size = size
        self.datadir = datadir
        self.enable_transforms = enable_transforms

        sortkey = lambda key: os.path.split(key)[-1]
        self.paths = sorted(make_dataset(datadir, fns), key=sortkey)
        if size is not None:
            self.paths = self.paths[:size]

        self.syn_model = ReflectionSythesis_1(kernel_sizes=[11], low_sigma=low_sigma, high_sigma=high_sigma, low_gamma=low_gamma, high_gamma=high_gamma)
        # self.reset(shuffle=False)

    # def reset(self, shuffle=True):
        # if shuffle:
            # random.shuffle(self.paths)
        # num_paths = len(self.paths) // 2
        # self.B_paths = self.paths[0:num_paths]
        # self.R_paths = self.paths[num_paths:2*num_paths]

    def data_synthesis(self, t_img, r_img):
        if self.enable_transforms:
            t_img, r_img = paired_data_transforms(t_img, r_img)
        syn_model = self.syn_model
        t_img, r_img, m_img = syn_model(t_img, r_img)
        
        B = to_tensor(t_img)
        R = to_tensor(r_img)
        M = to_tensor(m_img)

        return B, R, M
        
    def __getitem__(self, index):
        index_B = index % len(self.B_paths)
        index_R = index % len(self.R_paths)
        
        B_path = self.B_paths[index_B]
        R_path = self.R_paths[index_R]
        
        t_img = Image.open(B_path).convert('RGB')
        r_img = Image.open(R_path).convert('RGB')
    
        B, R, M = self.data_synthesis(t_img, r_img)

        fn = os.path.basename(B_path)
        return {'input': M, 'target_t': B, 'target_r': R, 'fn': fn}

    def __len__(self):
        if self.size is not None:
            return min(max(len(self.B_paths), len(self.R_paths)), self.size)
        else:
            return max(len(self.B_paths), len(self.R_paths))

class CEILTestDataset(BaseDataset):
    def __init__(self, datadir, fns=None, size=None, enable_transforms=False, unaligned_transforms=False, round_factor=1, flag=None):
        super(CEILTestDataset, self).__init__()
        self.size = size
        self.datadir = datadir
        self.fns = fns or os.listdir(join(datadir, 'blended'))
        self.enable_transforms = enable_transforms
        self.unaligned_transforms = unaligned_transforms
        self.round_factor = round_factor
        self.flag = flag

        # print(len(self.fns))
        
        if size is not None:
            self.fns = self.fns[:size]

    def __getitem__(self, index):
        fn = self.fns[index]
        
        t_img = Image.open(join(self.datadir, 'transmission_layer', fn)).convert('RGB')
        m_img = Image.open(join(self.datadir, 'blended', fn)).convert('RGB')
        
        if self.enable_transforms:
            t_img, m_img = paired_data_transforms(t_img, m_img, self.unaligned_transforms)

        B = to_tensor(t_img)
        M = to_tensor(m_img)

        dic =  {'input': M, 'target_t': B, 'fn': fn, 'real':True, 'target_r': B} # fake reflection gt 
        if self.flag is not None:
            dic.update(self.flag)
        return dic

    def __len__(self):
        if self.size is not None:
            return min(len(self.fns), self.size)
        else:
            return len(self.fns)

def get_uhd_transform(a1_img, a2_img, b_img, crop_size=256):
        # Random Crop
        w, h = a1_img.size
        random_w = random.randint(0, w-crop_size)
        random_h = random.randint(0, h-crop_size)

        a1_img = a1_img.crop((random_w, random_h, random_w+crop_size, random_h+crop_size))
        a2_img = a2_img.crop((random_w, random_h, random_w+crop_size, random_h+crop_size)) if a2_img else None
        b_img = b_img.crop((random_w, random_h, random_w+crop_size, random_h+crop_size))
        
        return to_tensor(a1_img), to_tensor(a2_img), to_tensor(b_img)

class UhdDataset(BaseDataset):
    def __init__(self, opt):
        super(UhdDataset, self).__init__()

        self.opt = opt

        self.trans_paths, self.reflect_paths, self.blended_paths, self.alpha_list = make_dataset_uhd(self.opt.dataroot,self.opt ,self.opt.phase ,self.opt.max_dataset_size)
        
        self.trans_size = len(self.trans_paths)  # get the size of dataset A1
        self.reflect_size = len(self.reflect_paths)  # get the size of dataset A2
        self.blended_size = len(self.blended_paths)

    def __getitem__(self, index):

        index = index % len(self.trans_paths)
        
        trans_path = self.trans_paths[index]
        reflect_path = self.reflect_paths[index]
        blended_path = self.blended_paths[index]
        
        trans_img = Image.open(trans_path).convert('RGB')
        reflect_img = Image.open(reflect_path).convert('RGB')
        blended_img = Image.open(blended_path).convert('RGB')

        T, R, B = get_uhd_transform(trans_img, reflect_img, blended_img)

        fn = os.path.basename(trans_path)

        return {'input': B, 'target_t': T, 'target_r': R, 'fn': fn}

    def __len__(self):
        length = min(max(self.trans_size, self.reflect_size, self.blended_size),self.opt.max_dataset_size)
        return length

class UhdTestDataset(BaseDataset):
    def __init__(self, opt):
        super(UhdTestDataset, self).__init__()
        self.opt = opt
        if self.opt.phase == 'train':
            self.testroot = os.path.abspath(os.path.join(self.opt.dataroot, '..', 'test/'))
        else:
            self.testroot = self.opt.dataroot
        # print(self.testroot)
        if self.opt.uhd == '4k':
            print('make_dataset_uhd')
            self.trans_paths, self.reflect_paths, self.blended_paths, self.alpha_list = make_dataset_uhd(self.testroot,self.opt, "test",self.opt.max_dataset_size)
        elif self.opt.uhd == '8k':
            print('make_dataset_uhd_crop 8k')
            self.trans_paths, self.reflect_paths, self.blended_paths = make_dataset_uhd_crop(self.testroot,self.opt,self.opt.max_dataset_size)
        else:
            print('make_dataset_uhd_crop 4kcrop')
            self.trans_paths, self.reflect_paths, self.blended_paths = make_dataset_uhd_crop(self.testroot,self.opt,self.opt.max_dataset_size)
        
        # self.dir_trans = os.path.join(self.testroot, 'transmission_layer')
        # self.dir_reflect = os.path.join(opt.testroot, 'test', 'reflection_layer')
        # self.dir_blended = os.path.join(self.testroot, 'blended')
        # print(self.trans_paths[0])

        # self.trans_paths = self.trans_paths[0:5]
        # self.reflect_paths = self.reflect_paths[0:5]
        # self.blended_paths = self.blended_paths[0:5]

        self.trans_size = len(self.trans_paths)  # get the size of dataset A1
        self.reflect_size = len(self.reflect_paths)  # get the size of dataset A2

        self.blended_size = len(self.blended_paths)
        print("Load %d dataset. UhdTestDataset " % self.blended_size)

        # print(self.trans_size)
        # print(self.blended_size)
        # assert len(self.trans_paths)==len(self.reflect_paths) ,'trans != reflect'
        assert len(self.trans_paths)==len(self.blended_paths) ,'trans != blended'
        
    def __getitem__(self, index):
        index = index % len(self.trans_paths)
        
        trans_path = self.trans_paths[index]
        reflect_path = self.reflect_paths[index]
        blended_path = self.blended_paths[index]
        
        trans_img = Image.open(trans_path).convert('RGB')
        reflect_img = Image.open(reflect_path).convert('RGB')
        blended_img = Image.open(blended_path).convert('RGB')

        # print(type(trans_img))

        if self.opt.phase == 'train':
            T, R, B = get_uhd_transform(trans_img, reflect_img, blended_img)
        else:
            T = F.to_tensor(trans_img)
            B = F.to_tensor(blended_img)
            
        # fn = os.path.basename(trans_path)
        fn = os.path.basename(trans_path)

        dic =  {'input': B, 'target_t': T, 'fn': fn, 'real':True, 'target_r': B} # fake reflection gt 

        return dic

    def __len__(self):
        length = min(max(self.trans_size, self.reflect_size, self.blended_size),self.opt.max_dataset_size)
        return length


class RealDataset(BaseDataset):
    def __init__(self, datadir, fns=None, size=None):
        super(RealDataset, self).__init__()
        self.size = size
        self.datadir = datadir
        self.fns = fns or os.listdir(join(datadir))
        
        if size is not None:
            self.fns = self.fns[:size]
        
    def __getitem__(self, index):
        fn = self.fns[index]
        B = -1
        
        m_img = Image.open(join(self.datadir, fn)).convert('RGB')

        M = to_tensor(m_img)
        data = {'input': M, 'target_t': B, 'fn': fn}
        return data

    def __len__(self):
        if self.size is not None:
            return min(len(self.fns), self.size)
        else:
            return len(self.fns)


class PairedCEILDataset(CEILDataset):
    def __init__(self, datadir, fns=None, size=None, enable_transforms=True, low_sigma=2, high_sigma=5):
        self.size = size
        self.datadir = datadir

        self.fns = fns or os.listdir(join(datadir, 'reflection_layer'))
        if size is not None:
            self.fns = self.fns[:size]

        self.syn_model = ReflectionSythesis_1(kernel_sizes=[11], low_sigma=low_sigma, high_sigma=high_sigma)
        self.enable_transforms = enable_transforms
        # self.reset()

    # def reset(self):
        # return

    def __getitem__(self, index):
        fn = self.fns[index]
        B_path = join(self.datadir, 'transmission_layer', fn)
        R_path = join(self.datadir, 'reflection_layer', fn)
        
        t_img = Image.open(B_path).convert('RGB')
        r_img = Image.open(R_path).convert('RGB')
    
        B, R, M = self.data_synthesis(t_img, r_img)

        data = {'input': M, 'target_t': B, 'target_r': R, 'fn': fn}
        # return M, B
        return data

    def __len__(self):
        if self.size is not None:
            return min(len(self.fns), self.size)
        else:
            return len(self.fns)


class FusionDataset(BaseDataset):
    def __init__(self, datasets, fusion_ratios=None):
        self.datasets = datasets
        self.size = sum([len(dataset) for dataset in datasets])
        self.fusion_ratios = fusion_ratios or [1./len(datasets)] * len(datasets)
        print('[i] using a fusion dataset: %d %s imgs fused with ratio %s' %(self.size, [len(dataset) for dataset in datasets], self.fusion_ratios))

    # def reset(self):
        # for dataset in self.datasets:
            # dataset.reset()

    def __getitem__(self, index):
        residual = 1
        for i, ratio in enumerate(self.fusion_ratios):
            if random.random() < ratio/residual or i == len(self.fusion_ratios) - 1:
                dataset = self.datasets[i]
                return dataset[index%len(dataset)]
            residual -= ratio
    
    def __len__(self):
        return self.size


def get_cdr_transform(a1_img, a2_img, b_img, crop_size=256):
        # Random Crop
        w, h = a1_img.size
        if w<256 or h<256:
            a1_img = a1_img.resize((crop_size,crop_size))
            a2_img = a2_img.resize((crop_size,crop_size))
            b_img = b_img.resize((crop_size,crop_size))
        else:
            random_w = random.randint(0, w-crop_size)
            random_h = random.randint(0, h-crop_size)

            a1_img = a1_img.crop((random_w, random_h, random_w+crop_size, random_h+crop_size))
            a2_img = a2_img.crop((random_w, random_h, random_w+crop_size, random_h+crop_size)) if a2_img else None
            b_img = b_img.crop((random_w, random_h, random_w+crop_size, random_h+crop_size))
            

        a1_img = F.to_tensor(a1_img)
        a2_img = F.to_tensor(a2_img)
        b_img = F.to_tensor(b_img)

        return a1_img, a2_img, b_img

class CdrTrainDataset(BaseDataset):
    '''
    # -----------------------------------------
    # Get T R B for CDR.
    # -----------------------------------------
    '''
    def __init__(self, opt):
        super(CdrTrainDataset, self).__init__()

        self.opt = opt

        trans_path = os.path.join(self.opt.dataroot, 'T')
        reflect_path = os.path.join(self.opt.dataroot, 'R')
        blended_path = os.path.join(self.opt.dataroot, 'M')

        self.trans_paths = sorted(make_dataset(trans_path))
        self.reflect_paths = sorted(make_dataset(reflect_path))
        self.blended_paths = sorted(make_dataset(blended_path))
            
        self.trans_size = len(self.trans_paths)
        self.reflect_size = len(self.reflect_paths)
        self.blended_size = len(self.blended_paths)

        print("Load %d CdrTrainDataset " % self.blended_size)

    def __getitem__(self, index):

        index = index % len(self.trans_paths)
        trans_path = self.trans_paths[index]
        reflect_path = self.reflect_paths[index]
        blended_path = self.blended_paths[index]

        trans_img = Image.open(trans_path).convert('RGB')
        reflect_img = Image.open(reflect_path).convert('RGB')
        blended_img = Image.open(blended_path).convert('RGB')

        T, R, B = get_cdr_transform(trans_img, reflect_img, blended_img)

        fn = os.path.basename(trans_path)

        return {'input': B, 'target_t': T, 'target_r': R, 'fn': fn}

    def __len__(self):
        return self.blended_size
        
class CdrTestDataset(BaseDataset):
    '''
    # -----------------------------------------
    # Get T R B for CDR.
    # -----------------------------------------
    '''
    def __init__(self, opt):
        super(CdrTestDataset, self).__init__()

        self.opt = opt

        trans_path = os.path.join(self.opt.dataroot, 'T')
        reflect_path = os.path.join(self.opt.dataroot, 'R')
        blended_path = os.path.join(self.opt.dataroot, 'M')

        self.trans_paths = sorted(make_dataset(trans_path))
        self.reflect_paths = sorted(make_dataset(reflect_path))
        self.blended_paths = sorted(make_dataset(blended_path))
            
        self.trans_size = len(self.trans_paths)
        self.reflect_size = len(self.reflect_paths)
        self.blended_size = len(self.blended_paths)

        print("Load %d CdrTrainDataset " % self.blended_size)

    def __getitem__(self, index):
            
        index = index % len(self.trans_paths)
    
        trans_path = self.trans_paths[index]
        # reflect_path = self.reflect_paths[index]
        blended_path = self.blended_paths[index]
        
        trans_img = Image.open(trans_path).convert('RGB')
        # reflect_img = Image.open(reflect_path).convert('RGB')
        blended_img = Image.open(blended_path).convert('RGB')

        T = F.to_tensor(trans_img)
        B = F.to_tensor(blended_img)
        # print("before padding:",B.shape)

        C, H, W = B.shape
        # 1280 640 320
        # 3840 1920 960
        if W < 640:
            paddingw = 960 - W
        elif W < 1280:
            paddingw = 1920 - W
        else:
            paddingw = 3840 - W

        # 720 360  
        # 2160 1080
        if H < 720:
            paddingh = 1080 - H
        else:
            paddingh = 2160 - H


        if paddingw % 2 != 0 and paddingh % 2 != 0:
            B = F.pad(B,(paddingw//2, paddingh//2, paddingw//2 + 1, paddingh//2 + 1), padding_mode="reflect")
        elif paddingw % 2 == 0 and paddingh % 2 != 0:
            B = F.pad(B,(paddingw//2, paddingh//2, paddingw//2, paddingh//2 + 1), padding_mode="reflect")
        elif paddingw % 2 != 0 and paddingh % 2 == 0:
            B = F.pad(B,(paddingw//2, paddingh//2, paddingw//2 + 1, paddingh//2), padding_mode="reflect")
        else:
            B = F.pad(B,(paddingw//2, paddingh//2), padding_mode="reflect")

        # print("after padding:",B.shape)

        fn = os.path.basename(trans_path)

        dic =  {'input': B, 'target_t': T, 'fn': fn, 'real':True, 'target_r': B} # fake reflection gt 

        return dic
        
    def __len__(self):
        return self.blended_size