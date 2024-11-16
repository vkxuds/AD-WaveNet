import os
import os.path
import numpy as np
import random
import h5py
import torch
import cv2
import glob
import torch.utils.data as udata
from torch.utils.data import Dataset, DataLoader
from skimage.metrics import peak_signal_noise_ratio
from PIL import Image

class MyValDataset(Dataset):
    def __init__(self, input_dir, target_dir):
        self.input_dir = input_dir
        self.target_dir = target_dir
        
        self.input_files = sorted(os.listdir(input_dir))
        self.target_files = sorted(os.listdir(target_dir))
        
        assert len(self.input_files) == len(self.target_files), "Input and target folders must have the same number of files"
    
    def __len__(self):
        return len(self.input_files)
    
    def __getitem__(self, idx):
        input_path = os.path.join(self.input_dir, self.input_files[idx])
        target_path = os.path.join(self.target_dir, self.target_files[idx])
        input_image = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
        target_image = cv2.imread(target_path, cv2.IMREAD_UNCHANGED)

        input_norm = np.float32(normalize(input_image))
        target_norm = np.float32(normalize(target_image))

        input_image = torch.tensor(input_norm, dtype=torch.float32).unsqueeze(0)  # 添加通道维度
        target_image = torch.tensor(target_norm, dtype=torch.float32).unsqueeze(0)  # 添加通道维度

        return input_image, target_image

def normalize(data):
    data=(2*(data-np.min(data))/(np.max(data)-np.min(data)))-1

    return data

def inormalize(data,org_data):
    min_data = np.min(org_data)
    max_data = np.max(org_data)
    original_data = ((data+1)*0.5)* (max_data - min_data) + min_data

    return original_data

def save_img(save_path, img):
    img = img.cpu()
    out_img = img[0].detach().numpy()
    out_img *= 255.0
    out_img = Image.fromarray(np.uint8(out_img[0]), mode='L')
    out_img.save(save_path)


def batch_PSNR(img, imclean, data_range, crop):
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    PSNR = 0
    for i in range(Img.shape[0]):
        PSNR += peak_signal_noise_ratio(Iclean[i,:,crop:Img.shape[2]-crop,crop:Img.shape[3]-crop], Img[i,:,crop:Img.shape[2]-crop,crop:Img.shape[3]-crop], data_range=data_range)
    return (PSNR/Img.shape[0])

def PSNR(img, imclean, data_range, crop):

    PSNR = 0
    PSNR += peak_signal_noise_ratio(imclean[crop:imclean.shape[2]-crop,crop:imclean.shape[3]-crop], img[crop:img.shape[2]-crop,crop:img.shape[3]-crop], data_range=data_range)
    return (PSNR)

def data_augmentation(image, mode):
    out = np.transpose(image, (1,2,0))
    if mode == 0:
        # original
        out = out
    elif mode == 1:
        # flip up and down
        out = np.flipud(out)
    elif mode == 2:
        # rotate counterwise 90 degree
        out = np.rot90(out)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        out = np.rot90(out)
        out = np.flipud(out)
    elif mode == 4:
        # rotate 180 degree
        out = np.rot90(out, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        out = np.rot90(out, k=2)
        out = np.flipud(out)
    elif mode == 6:
        # rotate 270 degree
        out = np.rot90(out, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        out = np.rot90(out, k=3)
        out = np.flipud(out)
    return np.transpose(out, (2,0,1))


def Im2Patch(img, win, stride=1):
    k = 0
    endc = img.shape[0]
    endw = img.shape[1]
    endh = img.shape[2]
    patch = img[:, 0:endw-win+0+1:stride, 0:endh-win+0+1:stride]
    TotalPatNum = patch.shape[1] * patch.shape[2]
    Y = np.zeros([endc, win*win,TotalPatNum], np.float32)
    for i in range(win):
        for j in range(win):
            patch = img[:,i:endw-win+i+1:stride,j:endh-win+j+1:stride]
            Y[:,k,:] = np.array(patch[:]).reshape(endc, TotalPatNum)
            k = k + 1
    return Y.reshape([endc, win, win, TotalPatNum])

def prepare_data(data_path, patch_size, stride, aug_times=1):
    # train
    print('process training input data')
    scales = [1]
    #scales = [1, 0.9, 0.8, 0.7]

    #files = glob.glob(os.path.join(data_path, 'final_kwave_0825/train/input', '*.tif'))
    #files_true = glob.glob(os.path.join(data_path, 'final_kwave_0825/train/true', '*.tif'))
    files = glob.glob(os.path.join(data_path, 'dataset_tomowave/fang_24_30_zero/train', '*.tif'))
    files_true = glob.glob(os.path.join(data_path, 'dataset_tomowave/fang24_true/train', '*.tif'))
    
    files.sort()
    files_true.sort()

    h5f = h5py.File('train_DnINN.h5', 'w')
    h5f_true = h5py.File('true_DnINN.h5', 'w')

    train_num = 0
    for i in range(len(files)):
        
        print(i,"/",len(files))
        img = cv2.imread(files[i],cv2.IMREAD_UNCHANGED)
        img_true = cv2.imread(files_true[i],cv2.IMREAD_UNCHANGED)
        img = np.expand_dims(img, axis=-1)
        img_true = np.expand_dims(img_true, axis=-1)

        h, w, c = img.shape
        
        for k in range(len(scales)):
            Img = cv2.resize(img, (int(h*scales[k]), int(w*scales[k])), interpolation=cv2.INTER_CUBIC)
            Img = np.expand_dims(Img, axis=-1)
            Img = np.expand_dims(Img[:,:,0].copy(), 0)
            Img = np.float32(normalize(Img))
            patches = Im2Patch(Img, win=patch_size, stride=stride)

            
            Img_true = cv2.resize(img_true, (int(h*scales[k]), int(w*scales[k])), interpolation=cv2.INTER_CUBIC)
            Img_true = np.expand_dims(Img_true, axis=-1)
            Img_true = np.expand_dims(Img_true[:,:,0].copy(), 0)
            Img_true = np.float32(normalize(Img_true))
            patches_true = Im2Patch(Img_true, win=patch_size, stride=stride)

            for n in range(patches.shape[3]):
                #print(range(len(files)),"//",range(len(scales)),'//',range(patches.shape[3]))
                data = patches[:, :, :, n].copy()
                data_true = patches_true[:, :, :, n].copy()

                num_random = np.random.randint(1, 8)
                data_aug = data_augmentation(data,num_random )
                data_aug_true = data_augmentation(data_true,num_random )

                h5f.create_dataset(str(train_num), data=data_aug)
                h5f_true.create_dataset(str(train_num), data=data_aug_true)

                train_num += 1
                '''for m in range(aug_times-1):

                    num_random_1 = np.random.randint(1,8)
                    data_aug = data_augmentation(data, num_random_1)
                    data_aug_true = data_augmentation(data_true,num_random_1)

                    h5f.create_dataset(str(train_num)+"_aug_%d" % (m+1), data=data_aug)
                    h5f_true.create_dataset(str(train_num)+"_aug_%d" % (m+1), data=data_aug_true)

                    train_num += 1'''
    h5f.close()
    h5f_true.close()

    print('training input and true set, # samples %d' % train_num)
    # val
    '''print('process validation data')
    files.clear()
    files = glob.glob(os.path.join(data_path, 'Set12', '*.png'))
    files.sort()
    h5f = h5py.File('val_DnINN.h5', 'w')
    val_num = 0
    for i in range(len(files)):
        img = cv2.imread(files[i])
        img = np.expand_dims(img[:,:,0], 0)
        img = np.float32(normalize(img))
        h5f.create_dataset(str(val_num), data=img)
        val_num += 1
    h5f.close()

    print('val set, # samples %d\n' % val_num)'''

def prepare_real_data(data_path, patch_size, stride, aug_times=1):
    # train
    print('process training true data')
    # scales = [1]
    scales = [1, 0.9, 0.8, 0.7]

    files = glob.glob(os.path.join(data_path, 'full_mice', '*.png'))
    files.sort()
    h5f = h5py.File('real_DnINN.h5', 'w')
    train_num = 0
    for i in range(len(files)):
        img = cv2.imread(files[i])
        h, w, c = img.shape
        for k in range(len(scales)):
            Img = cv2.resize(img, (int(h*scales[k]), int(w*scales[k])), interpolation=cv2.INTER_CUBIC)
            Img = np.expand_dims(Img[:,:,0].copy(), 0)
            Img = np.float32(normalize(Img))
            patches = Im2Patch(Img, win=patch_size, stride=stride)
            for n in range(patches.shape[3]):
                data = patches[:, :, :, n].copy()

                data_aug = data_augmentation(data, np.random.randint(1, 8))
                h5f.create_dataset(str(train_num), data=data_aug)
                train_num += 1
                for m in range(aug_times-1):
                    data_aug = data_augmentation(data, np.random.randint(1,8))
                    h5f.create_dataset(str(train_num)+"_aug_%d" % (m+1), data=data_aug)
                    train_num += 1
    h5f.close()
    
    print('training true set, # samples %d\n' % val_num)

class Dataset(udata.Dataset):
    def __init__(self,input,target, train=True):
        super(Dataset, self).__init__()
        self.train = train
        self.input = input
        self.target = target
        with h5py.File(self.input,'r') as h5f_input:
            self.keys = list(h5f_input.keys())
        #random.shuffle(self.keys)
        h5f_input.close()
    def __len__(self):
        return len(self.keys)
    def __getitem__(self, index):
        
        key = self.keys[index]
        with h5py.File(self.input,'r') as h5f_input, h5py.File(self.target,'r') as h5f_target:
            input_image = np.array(h5f_input[key])
            target_image = np.array(h5f_target[key])
        input_image=torch.Tensor(input_image)
        target_image=torch.Tensor(target_image)

        h5f_input.close()
        h5f_target.close()
        return input_image,target_image

